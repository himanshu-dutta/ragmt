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

import numpy as np
from typing import Tuple, List
from datasets import Dataset
from transformers.utils import requires_backends


class KnowledgeBaseIndex:
    def __init__(self, kb_ds_dir: str, kb_index_path: str) -> None:
        # requires_backends(self, ["datasets", "faiss"])
        self.dataset = Dataset.load_from_disk(kb_ds_dir)
        self.dataset.load_faiss_index("embedding", kb_index_path)
        self.dataset.set_format(
            "numpy", columns=["embedding"], output_all_columns=True, dtype="float32"
        )

    def get_top_docs(
        self, qs: np.ndarray, n_docs: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each query in the batch, retrieves `n_docs`. Args:
            qs (`np.ndarray` of shape `(batch_size, vector_size)`): An array of
            query vectors
            n_docs (`int`): Number of documents to retrieve per
            query
        Returns:
            `np.ndarray` of shape `(batch_size, n_docs)`: A tensor of indices of
            retrieved documents.
            `np.ndarray` of shape `(batch_size, vector_size)`: A tensor of vector representations of retrieved
            documents.
        """
        _, ids = self.dataset.search_batch("embedding", qs, n_docs)
        docs = [self.dataset[[idx for idx in indices if idx >= 0]] for indices in ids]
        vectors = [doc["embedding"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack(
                    [
                        vectors[i],
                        np.zeros((n_docs - len(vectors[i]), vectors[i].shape[1])),
                    ]
                )
        return np.array(ids), np.array(vectors)

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        """
        Returns a list of dictionaries, containing titles and text of the retrieved documents.

        Args:
            doc_ids (`np.ndarray` of shape `(batch_size, n_docs)`):
                A tensor of document indices.
        """
        return [self.dataset[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]
