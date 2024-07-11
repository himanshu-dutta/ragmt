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

import lightning as l
from typing import Any, List
from ragmt.model.integrator import BaseIntegrator
from ragmt.model.retriever import Retriever
from ragmt.model.generator import BaseGenerator
import ragmt.config as config
import torch
from torch.nn import CrossEntropyLoss
from argparse import ArgumentParser
from ragmt.evaluation import mt_evaluate

import enum


class RAGMTLossType(enum.Enum):
    GENRATOR_LOSS: int = 1
    GENERATOR_DOC_LOSS: int = 2
    GENERATOR_MLM_LOSS: int = 3
    RAGMT_LOSS: int = 4


class RAGMT(l.LightningModule):
    # TODO: Extend implementation for multiple <Retriever, Integrator> pairs.
    def __init__(
        self,
        retriever: Retriever,
        generator: BaseGenerator,
        integrator: BaseIntegrator,
        source_lang: str,
        target_lang: str,
        n_docs: int = config.N_DOCS,
        train_document_encoder: bool = False,
        use_retrieval: bool = True,
        learning_rate: int = config.LEARNING_RATE,
        adam_beta1: float = config.ADAM_BETA1,
        adam_beta2: float = config.ADAM_BETA2,
        adam_epsilon: float = config.ADAM_EPSILON,
        loss_type: RAGMTLossType = RAGMTLossType.RAGMT_LOSS,
    ):
        super().__init__()

        self.retriever = retriever
        self.generator = generator
        self.integrator = integrator

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.n_docs = n_docs
        self.train_document_encoder = train_document_encoder
        self.use_retrieval = use_retrieval
        self.loss_type = loss_type

        self.save_hyperparameters(
            "source_lang",
            "target_lang",
            "learning_rate",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "n_docs",
            "train_document_encoder",
            "use_retrieval",
        )

    def forward(
        self,
        source,
        source_input_ids=None,
        target_input_ids=None,
        source_masked=None,
        max_length: int = 1024,
        use_generate: bool = False,
        **generation_kwargs,
    ):
        (
            doc_dicts,
            similarity_score,
            _,
            _,
        ) = self.retriever(
            source,
            n_docs=self.n_docs,
            train_document_encoder=self.train_document_encoder,
        )

        docs = [doc["text"] for doc in doc_dicts]

        mt_input_ids, mt_attention_mask, mt_sources = self.integrator(
            source,
            docs,
            config.LANG_CONFIG[self.hparams.source_lang]["nllb_id"],
            max_length,
        )
        mlm_input_ids, mlm_attention_mask = None, None
        if source_masked != None:
            mlm_input_ids, mlm_attention_mask, _ = self.integrator(
                source_masked,
                docs,
                config.MASKED_LM_LANG_CODE,
                max_length,
            )
        if source_input_ids != None:
            source_input_ids = source_input_ids.repeat_interleave(self.n_docs, dim=0)

        if target_input_ids != None:
            target_input_ids = target_input_ids.repeat_interleave(self.n_docs, dim=0)
        # fmt: off
        (
            pred_mt_loss,
            pred_mt_logits,
            pred_mlm_loss,
            pred_mlm_logits,
        ) = self.generator(
            mt_input_ids.to(self.generator.device),
            mt_attention_mask.to(self.generator.device),
            target_input_ids.to(self.generator.device) if target_input_ids != None else None,
            mlm_input_ids.to(self.generator.device) if mlm_input_ids != None else None,
            mlm_attention_mask.to(self.generator.device) if mlm_attention_mask != None else None,
            source_input_ids.to(self.generator.device) if source_input_ids != None else None,
        )
        # fmt: on
        if use_generate:

            mt_outputs, decoded_outputs = self.generator.generate(
                mt_input_ids.to(self.generator.device),
                target_lang_code=config.LANG_CONFIG[self.hparams.target_lang][
                    "nllb_id"
                ],
                **generation_kwargs,
            )
            return (
                mt_outputs,
                decoded_outputs,
                mt_sources,
                pred_mt_loss,
                pred_mt_logits,
                pred_mlm_loss,
                pred_mlm_logits,
                docs,
                similarity_score,
            )

        return (
            pred_mt_loss,
            pred_mt_logits,
            pred_mlm_loss,
            pred_mlm_logits,
            docs,
            similarity_score,
        )

    def calcuate_loss(
        self,
        mt_logits,
        mt_target,
        mlm_logits,
        mlm_target,
        doc_similarity_score,
    ):

        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")

        mt_logits = mt_logits.to(mt_target.device)
        mlm_logits = mlm_logits.to(mt_target.device)

        mt_loss = (
            loss_fct(
                mt_logits.view(-1, self.generator.model.vocab_size), mlm_target.view(-1)
            )
            .reshape((mt_logits.shape[0], -1))
            .mean(dim=-1)
        )

        mlm_loss = (
            loss_fct(
                mlm_logits.view(-1, self.generator.model.vocab_size), mt_target.view(-1)
            )
            .reshape(mlm_logits.shape[0], -1)
            .mean(dim=-1)
        )

        if self.loss_type == RAGMTLossType.GENRATOR_LOSS:
            return mt_loss.mean()

        if self.loss_type == RAGMTLossType.GENERATOR_DOC_LOSS:
            return (doc_similarity_score.reshape((-1,)) * mt_loss).mean()

        if self.loss_type == RAGMTLossType.GENERATOR_MLM_LOSS:
            return mt_loss.mean() + mlm_loss.mean()

        return (doc_similarity_score.reshape((-1,)) * mt_loss).mean() + mlm_loss.mean()

    def training_step(self, batch: dict, batch_idx: int):
        source_input_ids = batch["source_input_ids"]
        target_input_ids = batch["target_input_ids"]
        source = batch["source"]
        source_masked = batch["source_masked"]
        batch_size = source_input_ids.shape[0]

        (
            mt_loss,
            mt_logits,
            mlm_loss,
            mlm_logits,
            docs,
            similarity_score,
        ) = self(
            source,
            source_input_ids,
            target_input_ids,
            source_masked,
            max_length=1024,
            use_generate=False,
        )

        loss = self.calcuate_loss(
            mt_logits,
            target_input_ids.repeat_interleave(self.n_docs, dim=0),
            mlm_logits,
            source_input_ids.repeat_interleave(self.n_docs, dim=0),
            similarity_score,
        )
        self.log(
            "train_mt_loss",
            mt_loss,
            on_epoch=True,
            on_step=True,
            batch_size=batch_size,
        )
        self.log(
            "train_mlm_loss",
            mlm_loss,
            on_epoch=True,
            on_step=True,
            batch_size=batch_size,
        )
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return mt_loss

    def validation_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        source_input_ids = batch["source_input_ids"]
        target_input_ids = batch["target_input_ids"]
        source = batch["source"]
        target = batch["target"]
        source_masked = batch["source_masked"]
        mt_targets = [sent for sent in target for _ in range(self.n_docs)]
        batch_size = source_input_ids.shape[0]

        (
            mt_outputs,
            decoded_outputs,
            mt_sources,
            mt_loss,
            mt_logits,
            mlm_loss,
            mlm_logits,
            docs,
            similarity_score,
        ) = self(
            source,
            source_input_ids,
            target_input_ids,
            source_masked,
            max_length=512,
            use_generate=True,
            do_sample=True,
            top_p=0.95,
            # num_beams=1,
        )

        metrics = mt_evaluate(decoded_outputs, mt_targets)
        metrics = {f"val_{m}": v for m, v in metrics.items()}

        loss = self.calcuate_loss(
            mt_logits,
            target_input_ids.repeat_interleave(self.n_docs, dim=0),
            mlm_logits,
            source_input_ids.repeat_interleave(self.n_docs, dim=0),
            similarity_score,
        )
        self.log(
            "val_mt_loss",
            mt_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val_mlm_loss",
            mlm_loss,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val_mean_similarity_score",
            similarity_score.mean(),
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        metrics.update(
            {
                "loss": mt_loss,
                "source": mt_sources,
                "target": mt_targets,
                "prediction": decoded_outputs,
            }
        )
        return metrics

    # def predict_step(
    #     self,
    # ): ...

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_epsilon,
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=5e-5)
        parser.add_argument("--adam_beta1", type=float, default=0.9)
        parser.add_argument("--adam_beta2", type=float, default=0.999)
        parser.add_argument("--adam_epsilon", type=float, default=1e-8)
        return parser
