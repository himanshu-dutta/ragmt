# Copyright (c) 2024 Himanshu Dutta
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


r"""
`build_kb` script is used to generate a knowledge base (KB) from a TSV file. 

The TSV file should have two columns, `title` and `text`. Out of the two, the
title field is optional.

The documents are encoded using a TextEncoder, for which, you can either provide
Huggingface repository name, or path to a pre-trained checkpoint in HuggingFace
format.

The documents are encoded and stored as arrow files, along with its `faiss`
index.
"""

import os
from functools import partial

import faiss
import torch
from datasets import load_dataset, Features, Value, Sequence
from dataclasses import dataclass, field
from transformers import (
    AutoModelForTextEncoding,
    AutoTokenizer,
    HfArgumentParser,
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
)
from typing import Union
import ragmt.config as config

torch.set_grad_enabled(False)


@dataclass
class RAGArgs:
    tsv_path: str = field(
        metadata={
            "help": "Path to a tab-separated tsv file with columns, 'title' (optional) and 'text'"
        }
    )
    output_dir: str = field(
        metadata={
            "help": "Path to a directory, where the Knowledge Base would be stored."
        }
    )
    document_encoder_path_or_name: str = field(
        default=config.DOCUMENT_ENCODER,
        metadata={
            "help": "The model used to encode the documents in the Knowledge Base. Typically, the model used here would be same as the one used in retriever."
        },
    )


@dataclass
class ProcessingArgs:
    batch_size: int = field(
        default=8,
        metadata={
            "help": "The batch size to use when computing the passages embeddings using the DPR context encoder."
        },
    )


@dataclass
class IndexArgs:
    d: int = field(
        default=config.FAISS_HNSW_D,
        metadata={
            "help": "The dimension of the embeddings to pass to the HNSW Faiss index."
        },
    )
    m: int = field(
        default=config.FAISS_HNSW_M,
        metadata={
            "help": (
                "The number of bi-directional links created for every new element during the HNSW index construction."
            )
        },
    )


def encode(
    documents: dict,
    document_encoder: Union[AutoModelForTextEncoding, DPRContextEncoder],
    document_tokenizer: Union[AutoTokenizer, DPRContextEncoderTokenizerFast],
) -> dict:
    documents["title"] = [
        title if title != None else "" for title in documents["title"]
    ]
    documents["text"] = [text if text != None else "" for text in documents["text"]]

    encodings = document_tokenizer(
        documents["title"],
        documents["text"],
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )
    embeddings = document_encoder(
        input_ids=encodings["input_ids"].to(document_encoder.device),
        attention_mask=encodings["attention_mask"].to(document_encoder.device),
        return_dict=True,
    ).pooler_output
    return {"embedding": embeddings.detach().cpu().numpy()}


def load_model_tokenizer(
    document_encoder_path_or_name: str = config.DOCUMENT_ENCODER,
) -> Union[
    Union[AutoModelForTextEncoding, DPRContextEncoder],
    Union[AutoTokenizer, DPRContextEncoderTokenizerFast],
]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, tokenizer = None, None
    if "dpr-ctx" in document_encoder_path_or_name:
        encoder = DPRContextEncoder.from_pretrained(document_encoder_path_or_name).to(
            device=device
        )
        tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
            document_encoder_path_or_name
        )
    else:
        encoder = AutoModelForTextEncoding.from_pretrained(
            document_encoder_path_or_name
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(document_encoder_path_or_name)
    return encoder, tokenizer


def build_kb(
    tsv_path: str,
    output_dir: str,
    document_encoder_path_or_name: str = config.DOCUMENT_ENCODER,
    batch_size: int = 8,
    d: int = config.FAISS_HNSW_D,
    m: int = config.FAISS_HNSW_M,
) -> None:
    assert os.path.exists(tsv_path), f"tsv file {tsv_path} doesn't exist"
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset(
        "csv",
        data_files=[tsv_path],
        split="train",
        delimiter="\t",
        column_names=["title", "text"],
    )

    document_encoder, document_tokenizer = load_model_tokenizer(
        document_encoder_path_or_name
    )
    dataset = dataset.map(
        partial(
            encode,
            document_encoder=document_encoder,
            document_tokenizer=document_tokenizer,
        ),
        batched=True,
        batch_size=batch_size,
        features=Features(
            {
                "title": Value("string"),
                "text": Value("string"),
                "embedding": Sequence(Value("float32")),
            }
        ),
    )
    dataset.save_to_disk(os.path.join(output_dir, "dataset"))

    index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embedding", custom_index=index)
    dataset.get_index("embedding").save(
        os.path.join(output_dir, "index.hnswflat.faiss")
    )


if __name__ == "__main__":
    parser = HfArgumentParser((RAGArgs, ProcessingArgs, IndexArgs))
    rag_args, processing_args, index_args = parser.parse_args_into_dataclasses()

    build_kb(
        tsv_path=rag_args.tsv_path,
        output_dir=rag_args.output_dir,
        document_encoder_path_or_name=rag_args.document_encoder_path_or_name,
        batch_size=processing_args.batch_size,
        d=index_args.d,
        m=index_args.m,
    )
