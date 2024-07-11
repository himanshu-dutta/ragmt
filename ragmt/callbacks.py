import lightning as l
from lightning.pytorch.loggers import WandbLogger
import random
import pandas as pd


class WandBLogPredictionSamplesCallback(l.Callback):
    def __init__(
        self, wandb_logger: WandbLogger, num_samples: int, log_freq: int = 1
    ) -> None:
        super().__init__()
        self.wandb_logger = wandb_logger
        self.sample_idxs = None
        self.max_index = None
        self.log_freq = log_freq
        self.num_samples = num_samples
        self.epoch = None

    def generate_sample_indices(self, st: int, en: int, num_samples):
        indices = list(range(st, en + 1))
        random.shuffle(indices)
        return indices[:num_samples]

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # logging the output of the first batch
        # every `self.log_freq` epoch

        if self.epoch != None and self.epoch == trainer.current_epoch:
            return

        self.epoch = trainer.current_epoch
        if self.epoch % self.log_freq != 0:
            return

        if batch_idx == 0:
            source = outputs["source"]
            target = outputs["target"]
            prediction = outputs["prediction"]

            if self.sample_idxs == None or self.max_index != len(source):
                self.max_index = len(source)
                if self.num_samples > self.max_index:
                    self.sample_idxs = list(range(0, self.max_index))
                else:
                    self.sample_idxs = self.generate_sample_indices(
                        0, self.max_index, self.num_samples
                    )

            source_sample = [source[idx] for idx in self.sample_idxs]
            target_sample = [target[idx] for idx in self.sample_idxs]
            prediction_sample = [prediction[idx] for idx in self.sample_idxs]
            epoch = [self.epoch for _ in self.sample_idxs]

            data = {
                "source": source_sample,
                "target": target_sample,
                "prediction": prediction_sample,
                "epoch": epoch,
            }
            df = pd.DataFrame(data)
            self.wandb_logger.log_table(
                key="val_prediction",
                dataframe=df,
            )
