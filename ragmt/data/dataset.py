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

import os

from torch.utils.data import Dataset
from tqdm import tqdm
from ragmt.data.utils import NER
import ragmt.config as config
from transformers import NllbTokenizer
import math


class RAGMTDataset(Dataset):
    def __init__(
        self,
        source_text,
        target_text=None,
        source_lang="german",
        target_lang="english",
        generate_masked=False,
        mask_token=config.MASK_TOKEN,
    ):
        assert target_text == None or len(source_text) == len(
            target_text
        ), "either target text should be None or number of source sentences and target sentences should be same"
        self.source_text = source_text
        self.target_text = target_text
        self.mask_token = mask_token
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.source_masked_text = (
            self._run_ner(self.source_text, lang=source_lang)
            if generate_masked
            else None
        )

    @classmethod
    def from_file(cls, source_file, target_file=None, **kwargs):
        source_text = None
        with open(source_file, "r") as fp:
            source_text = [ln.strip() for ln in fp.readlines()]

        target_text = None
        if target_file != None:
            with open(target_file, "r") as fp:
                target_text = [ln.strip() for ln in fp.readlines()]
        return cls(source_text, target_text, **kwargs)

    def _run_ner(self, texts, lang="german", batch_size=config.NER_BATCH_SIZE):
        ner = NER(lang)
        masked_text = list()
        for bidx in tqdm(
            range(0, len(texts), batch_size),
            total=math.ceil(len(texts) / batch_size),
        ):
            mtxt = texts[bidx : bidx + batch_size]
            entities = ner(mtxt)
            for idx in range(len(mtxt)):
                for ent in entities[idx]:
                    mtxt[idx] = mtxt[idx].replace(ent, self.mask_token)
                masked_text.append(mtxt[idx])
        return masked_text

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, idx):
        return {
            "source": self.source_text[idx],
            "target": self.target_text[idx] if self.target_text else None,
            "source_masked": (
                self.source_masked_text[idx] if self.source_masked_text else None
            ),
        }


def tokenize_fn(
    datapoints,
    tokenizer: NllbTokenizer,
    src_lang_code,
    tgt_lang_code,
    max_input_length=config.GENERATOR_OUTPUT_MAX_LENGTH,
    max_target_length=config.GENERATOR_OUTPUT_MAX_LENGTH,
):
    sources = [datapoint["source"] for datapoint in datapoints]
    tokenizer.src_lang = src_lang_code
    sources_tokenized = tokenizer(
        sources,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    targets = [
        datapoint["target"] if datapoint["target"] != None else ""
        for datapoint in datapoints
    ]
    tokenizer.src_lang = tgt_lang_code
    targets_tokenized = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    sources_masked = [
        datapoint["source_masked"] if datapoint["source_masked"] != None else ""
        for datapoint in datapoints
    ]
    tokenizer.src_lang = src_lang_code
    sources_masked_tokenized = tokenizer(
        sources_masked,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    sources_tokenized["input_ids"][
        sources_tokenized["input_ids"] == tokenizer.pad_token_id
    ] = -100
    targets_tokenized["input_ids"][
        targets_tokenized["input_ids"] == tokenizer.pad_token_id
    ] = -100

    model_inputs = {
        "source_input_ids": sources_tokenized["input_ids"],
        "source_attention_mask": sources_tokenized["attention_mask"],
        "target_input_ids": targets_tokenized["input_ids"],
        "target_attention_mask": targets_tokenized["attention_mask"],
        "source_masked_input_ids": sources_masked_tokenized["input_ids"],
        "source_masked_attention_mask": sources_masked_tokenized["attention_mask"],
        "source": sources,
        "target": targets,
        "source_masked": sources_masked,
    }

    return model_inputs


def load_datasets(
    dataset_dir: str,
    source_lang: str,
    target_lang: str,
    generate_masked=False,
    mask_token=config.MASK_TOKEN,
    splits=["train", "val", "test"],
):
    """
    Expects the files for each split to be in the format: `<split>.<source_lang_code>` and `<split>.<target_lang_code>`
    Ex: `train.en` and `train.de`.
    """
    source_lang_code = config.LANG_CONFIG[source_lang]["dataset_ext"]
    target_lang_code = config.LANG_CONFIG[target_lang]["dataset_ext"]
    datasets = dict()

    for split in splits:
        source_path = os.path.join(dataset_dir, f"{split}.{source_lang_code}")
        target_path = os.path.join(dataset_dir, f"{split}.{target_lang_code}")
        datasets[split] = RAGMTDataset.from_file(
            source_path,
            target_path,
            source_lang=source_lang,
            target_lang=target_lang,
            generate_masked=generate_masked,
            mask_token=mask_token,
        )

    return datasets
