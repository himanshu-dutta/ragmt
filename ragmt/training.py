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

os.environ["TOKENIZERS_PARALLELISM"] = "false"


from argparse import ArgumentParser
from functools import partial

import torch
import lightning as l
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import ragmt.config as config

from ragmt.model.ragmt import RAGMT, RAGMTLossType
from ragmt.model.utils import (
    load_nllb_model_tokenizer,
    load_document_query_encoder_model_tokenizer,
    load_ragmt_from_base_modules,
)
from ragmt.callbacks import WandBLogPredictionSamplesCallback

from ragmt.model.retriever import Retriever
from ragmt.model.integrator import TriplesIntegrator
from ragmt.model.generator import NllbMultitaskModel

from ragmt.data.dataset import tokenize_fn, load_datasets
from ragmt.data.knowledgebase import KnowledgeBaseIndex
import gc


def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


def train():
    cleanup()
    l.seed_everything(42)
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--query_encoder_model_path", type=str)
    parser.add_argument("--generator_model_path", type=str)
    parser.add_argument("--document_encoder_model_path", type=str, default=None)

    parser.add_argument("--kb_dataset_dir", type=str)
    parser.add_argument("--kb_index_path", type=str)

    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--source_lang", type=str)
    parser.add_argument("--target_lang", type=str)

    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--use_fp16", action="store_true")

    parser.add_argument("--train_document_encoder", action="store_true")
    parser.add_argument("--pretrained_ragmt_path", type=str, default=None)
    parser.add_argument("--use_retrieval", action="store_true")

    parser = RAGMT.add_model_specific_args(parser)
    args = parser.parse_args()

    model = load_ragmt_from_base_modules(
        query_encoder_model_path=args.query_encoder_model_path,
        generator_model_path=args.generator_model_path,
        kb_dataset_dir=args.kb_dataset_dir,
        kb_index_path=args.kb_index_path,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        document_encoder_model_path=args.document_encoder_model_path,
        train_document_encoder=args.train_document_encoder,
        use_retrieval=args.use_retrieval,
        pretrained_path=args.pretrained_ragmt_path,
        loss_type=RAGMTLossType.RAGMT_LOSS,
    )

    datasets = load_datasets(
        args.dataset_dir,
        args.source_lang,
        args.target_lang,
        generate_masked=True,
        mask_token=config.MASK_TOKEN,
        splits=["train", "val"],
    )

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.train_batch_size,
            collate_fn=partial(
                tokenize_fn,
                tokenizer=model.generator.tokenizer,
                src_lang_code=config.LANG_CONFIG[args.source_lang]["nllb_id"],
                tgt_lang_code=config.LANG_CONFIG[args.target_lang]["nllb_id"],
            ),
            num_workers=32,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.val_batch_size,
            collate_fn=partial(
                tokenize_fn,
                tokenizer=model.generator.tokenizer,
                src_lang_code=config.LANG_CONFIG[args.source_lang]["nllb_id"],
                tgt_lang_code=config.LANG_CONFIG[args.target_lang]["nllb_id"],
            ),
            num_workers=32,
        ),
    }

    logger = WandbLogger(project="ragmt", name=args.experiment_name)
    trainer = l.Trainer(
        devices=args.num_gpus,
        max_epochs=30,
        accelerator="auto",
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[
            WandBLogPredictionSamplesCallback(
                logger,
                num_samples=32,
                log_freq=1,
            )
        ],
        precision=16 if args.use_fp16 else None,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    trainer.fit(
        model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
        ckpt_path=args.pretrained_ragmt_path,
    )
    trainer.save_checkpoint(
        "/home/artifacts/model/kg-domainadaptation-koran-dpr_ctx_encoder-nllb200_600M-En-De"
    )


if __name__ == "__main__":
    train()
