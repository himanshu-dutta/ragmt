import math
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import json
import argparse


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/mrebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/mrebel-large")

if torch.cuda.is_available():
    model.to(torch.device("cuda"))


# from https://huggingface.co/Babelscape/rebel-large
def extract_relations_from_model_output(text):
    triplets = []
    relation = ""
    text = text.strip()
    current = "x"
    subject, relation, object_, object_type, subject_type = "", "", "", "", ""

    for token in (
        text.replace("<s>", "")
        .replace("<pad>", "")
        .replace("</s>", "")
        .replace("tp_XX", "")
        .replace("__en__", "")
        .split()
    ):
        if token == "<triplet>" or token == "<relation>":
            current = "t"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "head_type": subject_type,
                        "type": relation.strip(),
                        "tail": object_.strip(),
                        "tail_type": object_type,
                    }
                )
                relation = ""
            subject = ""
        elif token.startswith("<") and token.endswith(">"):
            if current == "t" or current == "o":
                current = "s"
                if relation != "":
                    triplets.append(
                        {
                            "head": subject.strip(),
                            "head_type": subject_type,
                            "type": relation.strip(),
                            "tail": object_.strip(),
                            "tail_type": object_type,
                        }
                    )
                object_ = ""
                subject_type = token[1:-1]
            else:
                current = "o"
                object_type = token[1:-1]
                relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if (
        subject != ""
        and relation != ""
        and object_ != ""
        and object_type != ""
        and subject_type != ""
    ):
        triplets.append(
            {
                "head": subject.strip(),
                "head_type": subject_type,
                "type": relation.strip(),
                "tail": object_.strip(),
                "tail_type": object_type,
            }
        )
    return triplets


# knowledge base class
class KB:
    def __init__(self):
        self.entities = set()
        self.relations = list()

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def merge_relations(self, r1):
        r2 = [r for r in self.relations if self.are_relations_equal(r1, r)][0]
        spans_to_add = [
            span for span in r1["meta"]["spans"] if span not in r2["meta"]["spans"]
        ]
        r2["meta"]["spans"] += spans_to_add

    def merge_kb(self, kb2):
        self.entities.union(kb2.entities)
        for r in kb2.relations:
            self.add_relation(r)

    def add_relation(self, r):
        self.entities.add(r["head"])
        self.entities.add(r["tail"])

        if not self.exists_relation(r):
            self.relations.append(r)
        else:
            self.merge_relations(r)

    def print(self):
        print("Entities:")
        for e in self.entities:
            print(f"  {e}")
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")

    def save_dict(self, path: str, additional_info: dict = {}):
        kg_dump = {
            "entities": list(self.entities),
            "relations": self.relations,
        }
        kg_dump.update(additional_info)

        with open(path, "w") as fp:
            json.dump(kg_dump, fp)


# build a knowledge base from text
def from_small_text_to_kb(text, verbose=False):
    kb = KB()

    # Tokenizer text
    model_inputs = tokenizer(
        text, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    if verbose:
        print(f"Num tokens: {len(model_inputs['input_ids'][0])}")

    # Generate
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 3,
        "forced_bos_token_id": None,
    }

    generated_tokens = model.generate(
        **model_inputs,
        **gen_kwargs,
    )
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # create kb
    for sentence_pred in decoded_preds:
        relations = extract_relations_from_model_output(sentence_pred)
        for r in relations:
            kb.add_relation(r)

    return kb


# extract relations for each span and put them together in a knowledge base
def from_text_to_kb(text, span_length=128, verbose=False):
    # tokenize whole text
    inputs = tokenizer(
        text, max_length=len(text), padding=True, truncation=False, return_tensors="pt"
    )

    # compute span boundaries
    num_tokens = len(inputs["input_ids"][0])
    if verbose:
        print(f"Input has {num_tokens} tokens")
    num_spans = math.ceil(num_tokens / span_length)
    if verbose:
        print(f"Input has {num_spans} spans")
    overlap = math.ceil((num_spans * span_length - num_tokens) / max(num_spans - 1, 1))
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append(
            [start + span_length * i, start + span_length * (i + 1)]
        )
        start -= overlap
    if verbose:
        print(f"Span boundaries are {spans_boundaries}")

    # transform input with spans
    tensor_ids = [
        inputs["input_ids"][0][boundary[0] : boundary[1]]
        for boundary in spans_boundaries
    ]
    tensor_masks = [
        inputs["attention_mask"][0][boundary[0] : boundary[1]]
        for boundary in spans_boundaries
    ]
    inputs = {
        "input_ids": torch.stack(tensor_ids),
        "attention_mask": torch.stack(tensor_masks),
    }

    # generate relations
    num_return_sequences = 3
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences,
        "forced_bos_token_id": None,
    }
    generated_tokens = model.generate(
        inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        decoder_start_token_id=tokenizer.convert_tokens_to_ids("tp_XX"),
        **gen_kwargs,
    )

    # decode relations
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # create kb
    kb = KB()
    i = 0
    for sentence_pred in decoded_preds:
        current_span_index = i // num_return_sequences
        relations = extract_relations_from_model_output(sentence_pred)
        for relation in relations:
            relation["meta"] = {"spans": [spans_boundaries[current_span_index]]}
            kb.add_relation(relation)
        i += 1

    return kb


# # from KB to HTML visualization
# def save_network_html(kb, filename="network.html"):
#     # create network
#     net = Network(directed=True, width="auto", height="700px", bgcolor="#eeeeee")

#     # nodes
#     color_entity = "#00FF00"
#     for e in kb.entities:
#         net.add_node(e, shape="circle", color=color_entity)

#     # edges
#     for r in kb.relations:
#         net.add_edge(r["head"], r["tail"],
#                     title=r["type"], label=r["type"])

#     # save network
#     net.repulsion(
#         node_distance=200,
#         central_gravity=0.2,
#         spring_length=200,
#         spring_strength=0.05,
#         damping=0.09
#     )
#     net.set_edge_smooth('dynamic')
#     net.show(filename)


def parse_raw_data(data, cascade_documents: bool = False):
    docs = data
    if cascade_documents:
        docs = " ".join(data)
    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("KG Extraction")
    parser.add_argument("-d", "--data-path", type=str)
    parser.add_argument("-o", "--output-dir", type=str)
    parser.add_argument("-c", "--checkpoint-interval", type=int, default=10)
    args = parser.parse_args()

    checkpoint_count = 10
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    print("Starting execution...")

    with open(args.data_path, "r") as fp:
        raw_data = fp.readlines()
    data = parse_raw_data(raw_data, cascade_documents=False)
    print("Data parsed successfully")

    for idx, doc in enumerate(data):
        kb_ = from_text_to_kb(doc, verbose=False)
        kb_save_path = args.output_dir + f"/{idx}.json"
        net_save_path = args.output_dir + f"/{idx}_network.html"
        kb_.save_dict(kb_save_path)

        if idx % checkpoint_count == 0:
            print(f"Generated KG till: {idx+1} documents.")
