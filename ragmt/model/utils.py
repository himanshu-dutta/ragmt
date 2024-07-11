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

from transformers import (
    NllbTokenizer,
    M2M100ForConditionalGeneration,
    AutoModelForTextEncoding,
    AutoTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
)
from typing import Union

import ragmt.config as config

from ragmt.model.ragmt import RAGMT, RAGMTLossType
from ragmt.model.retriever import Retriever
from ragmt.model.integrator import TriplesIntegrator
from ragmt.model.generator import NllbMultitaskModel

from ragmt.data.knowledgebase import KnowledgeBaseIndex


def load_nllb_model_tokenizer(
    pretrained_model_name_or_path,
    masked_lm_lang_code=config.MASKED_LM_LANG_CODE,
):
    r"""
    We add the token `config.MASKED_LM_LANG_CODE` so as to distinguish between translation and Masked LM task.
    """
    pretrained_tokenizer = NllbTokenizer.from_pretrained(
        pretrained_model_name_or_path,
    )
    model = M2M100ForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path
    )

    if masked_lm_lang_code not in pretrained_tokenizer.additional_special_tokens:
        tokenizer = NllbTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            additional_special_tokens=pretrained_tokenizer.additional_special_tokens
            + [masked_lm_lang_code],
        )
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def load_document_query_encoder_model_tokenizer(
    document_encoder_path_or_name: str = config.DOCUMENT_ENCODER,
) -> Union[
    Union[AutoModelForTextEncoding, DPRContextEncoder],
    Union[AutoTokenizer, DPRContextEncoderTokenizerFast],
]:
    encoder, tokenizer = None, None
    if "dpr-ctx" in document_encoder_path_or_name:
        encoder = DPRContextEncoder.from_pretrained(document_encoder_path_or_name)
        tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
            document_encoder_path_or_name
        )
    else:
        encoder = AutoModelForTextEncoding.from_pretrained(
            document_encoder_path_or_name
        )
        tokenizer = AutoTokenizer.from_pretrained(document_encoder_path_or_name)
    return encoder, tokenizer


def load_ragmt_from_base_modules(
    query_encoder_model_path: str,
    generator_model_path: str,
    kb_dataset_dir: str,
    kb_index_path: str,
    source_lang: str,
    target_lang: str,
    document_encoder_model_path=None,
    train_document_encoder: bool = False,
    use_retrieval: bool = True,
    loss_type: RAGMTLossType = RAGMTLossType.RAGMT_LOSS,
    pretrained_path: str = None,
) -> RAGMT:
    query_encoder, query_tokenizer = load_document_query_encoder_model_tokenizer(
        query_encoder_model_path
    )
    document_encoder, document_tokenizer = None, None
    if train_document_encoder:
        assert (
            document_encoder_model_path != None
        ), "to train document encoder, path to pretrained model must be provided"
        document_encoder, document_tokenizer = (
            load_document_query_encoder_model_tokenizer(document_encoder_model_path)
        )

    generator_model, generator_tokenizer = load_nllb_model_tokenizer(
        generator_model_path,
        config.MASKED_LM_LANG_CODE,
    )

    model = RAGMT(
        retriever=Retriever(
            query_encoder,
            query_tokenizer,
            KnowledgeBaseIndex(kb_dataset_dir, kb_index_path),
            document_encoder,
            document_tokenizer,
        ),
        generator=NllbMultitaskModel(
            model=generator_model, tokenizer=generator_tokenizer
        ),
        integrator=TriplesIntegrator(generator_tokenizer),
        source_lang=source_lang,
        target_lang=target_lang,
        train_document_encoder=train_document_encoder,
        use_retrieval=use_retrieval,
        loss_type=loss_type,
    )

    if pretrained_path:
        model = model.load_from_checkpoint(pretrained_path)
    return model
