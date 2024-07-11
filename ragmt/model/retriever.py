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

from typing import List, Union
from transformers import (
    AutoModelForTextEncoding,
    AutoTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
)
import numpy as np
from ragmt.data.knowledgebase import KnowledgeBaseIndex
import ragmt.config as config
import torch
import lightning as l


class Retriever(l.LightningModule):
    def __init__(
        self,
        query_encoder: Union[AutoModelForTextEncoding, DPRContextEncoder],
        query_tokenizer: Union[AutoTokenizer, DPRContextEncoderTokenizerFast],
        knowledge_base: KnowledgeBaseIndex,
        document_encoder: Union[
            AutoModelForTextEncoding, DPRContextEncoder, None
        ] = None,
        document_tokenizer: Union[
            AutoTokenizer, DPRContextEncoderTokenizerFast, None
        ] = None,
    ):
        super().__init__()
        self.query_encoder = query_encoder
        self.query_tokenizer = query_tokenizer
        self.knowledge_base = knowledge_base
        self.document_encoder = document_encoder
        self.document_tokenizer = document_tokenizer

    def encode_query(
        self,
        query: List[str],
    ) -> dict:
        encodings = self.query_tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).to(self.query_encoder.device)
        embeddings = self.query_encoder(
            input_ids=encodings["input_ids"].to(self.query_encoder.device),
            attention_mask=encodings["attention_mask"].to(self.query_encoder.device),
            return_dict=True,
        ).pooler_output
        return embeddings

    def encode_doc(self, titles: List[str], texts: List[str]) -> dict:
        encodings = self.document_tokenizer(
            titles,
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).to(self.document_encoder.device)
        embeddings = self.document_encoder(
            input_ids=encodings["input_ids"].to(self.document_encoder.device),
            attention_mask=encodings["attention_mask"].to(self.document_encoder.device),
            return_dict=True,
        ).pooler_output
        return embeddings

    def retrieve(self, query: np.ndarray, n_docs: int):
        doc_ids, doc_vecs = self.knowledge_base.get_top_docs(query, n_docs)
        doc_dicts = self.knowledge_base.get_doc_dicts(doc_ids)
        return doc_ids, doc_vecs, doc_dicts

    def __call__(
        self,
        query: Union[str, List[str]],
        n_docs: int = config.N_DOCS,
        train_document_encoder: bool = False,
    ):
        if isinstance(query, str):
            query = [query]
        if train_document_encoder:
            assert self.document_encoder != None and self.document_tokenizer != None

        query = self.encode_query(query)
        doc_ids, doc_vecs, doc_dicts = self.retrieve(
            query.detach().cpu().numpy(),
            n_docs,
        )

        if train_document_encoder:
            retrieved_doc_text = []
            retrieved_doc_title = []

            for b_idx in range(len(doc_dicts)):
                for doc_idx in range(n_docs):
                    retrieved_doc_text.append(doc_dicts[b_idx]["text"][doc_idx])
                    retrieved_doc_title.append(doc_dicts[b_idx]["title"][doc_idx])

            doc_encs = self.encode_doc(
                retrieved_doc_title,
                retrieved_doc_text,
            ).reshape(len(doc_dicts), n_docs, -1)
        else:
            doc_encs = torch.tensor(doc_vecs, device=query.device)

        similarity_score = (
            torch.bmm(doc_encs, query.unsqueeze(2)).squeeze(-1).softmax(-1)
        )

        return (doc_dicts, similarity_score, doc_ids, doc_vecs)

    def save_model(self): ...

    def load_model(self): ...
