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

from transformers import NllbTokenizer
from typing import List, Union


class BaseIntegrator:
    def __call__(
        self,
        text: Union[str, List[str]],
        docs: Union[List[str], List[List[str]]],
    ):
        raise NotImplementedError()


class NoIntegration(BaseIntegrator): ...


class TriplesIntegrator(BaseIntegrator):
    r"""
    Implements integration of knowledge graph triples with source sentence
    Source: src
    Docs: <t1, t2, ..., tk>
    Format: t1 <LANG_CODE> src
    """

    def __init__(
        self,
        tokenizer: NllbTokenizer,
        triples_seperator_token: str = None,
    ):
        self.tokenizer = tokenizer
        self.triples_seperator_token = (
            triples_seperator_token
            if triples_seperator_token != None
            else tokenizer.sep_token
        )

    def __call__(
        self,
        text: Union[str, List[str]],
        docs: Union[List[str], List[List[str]]],
        lang_code: str,
        max_length: int,
    ):
        """
        Use the `lang` argument to control the language, as well as masking task
        """
        if isinstance(text, str):
            text = [text]
            docs = [docs]

        inputs = list()
        for t, ds in zip(text, docs):
            for d in ds:
                inp = f"{d} {lang_code} {t}"
                inputs.append(inp)

        encodings = self.tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"][:, 1:]
        attention_mask = encodings["attention_mask"][:, 1:]
        return input_ids, attention_mask, inputs
