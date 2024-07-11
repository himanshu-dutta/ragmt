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
from transformers import M2M100ForConditionalGeneration, AutoTokenizer
import torch


class BaseGenerator(l.LightningModule):
    def forward(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        mt_target_input_ids: torch.Tensor = None,
        source_masked_input_ids: torch.Tensor = None,
        source_masked_attention_mask: torch.Tensor = None,
        mlm_target_input_ids: torch.Tensor = None,
    ):
        raise NotImplementedError()


class NllbMultitaskModel(BaseGenerator):
    def __init__(self, model: M2M100ForConditionalGeneration, tokenizer: AutoTokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        mt_target_input_ids: torch.Tensor = None,
        source_masked_input_ids: torch.Tensor = None,
        source_masked_attention_mask: torch.Tensor = None,
        mlm_target_input_ids: torch.Tensor = None,
    ):
        pred_mt_target = self.model(
            source_input_ids,
            source_attention_mask,
            labels=mt_target_input_ids,
        )
        pred_mlm_target = None
        if (
            source_masked_input_ids != None
            and source_masked_attention_mask != None
            and mlm_target_input_ids != None
        ):
            pred_mlm_target = self.model(
                source_masked_input_ids,
                source_masked_attention_mask,
                labels=mlm_target_input_ids,
            )
        return (
            pred_mt_target.loss,
            pred_mt_target.logits,
            pred_mlm_target.loss if pred_mlm_target else None,
            pred_mlm_target.logits if pred_mlm_target else None,
        )

    def generate(
        self,
        source_input_ids: torch.Tensor,
        target_lang_code: str,
        **generate_kwargs,
    ):
        source_input_ids[source_input_ids == -100] = self.tokenizer.pad_token_id
        mt_outputs = self.model.generate(
            source_input_ids,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang_code],
            **generate_kwargs,
        )
        mt_outputs[mt_outputs == -100] = self.tokenizer.pad_token_id

        decoded_outputs = self.tokenizer.batch_decode(
            mt_outputs.tolist(), skip_special_tokens=True
        )
        return (mt_outputs, decoded_outputs)
