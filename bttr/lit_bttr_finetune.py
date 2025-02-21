import zipfile

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor

from bttr.datamodule import Batch, vocab
from bttr.model.bttr_finetune import BTTR
from bttr.utils import ExpRateRecorder, Hypothesis, ce_loss, to_bi_tgt_out

from pytorch_lightning.callbacks import BaseFinetuning

class LitBTTR(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        # training
        learning_rate: float,
        finetune_learning_rate: float,
        patience: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.bttr = BTTR(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.exprate_recorder = ExpRateRecorder()

        self.test_outputs = []

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.bttr(img, img_mask, tgt)

    def beam_search(
        self,
        img: FloatTensor,
        beam_size: int = 10,
        max_len: int = 200,
        alpha: float = 1.0,
    ) -> str:
        """for inference, one image at a time

        Parameters
        ----------
        img : FloatTensor
            [1, h, w]
        beam_size : int, optional
            by default 10
        max_len : int, optional
            by default 200
        alpha : float, optional
            by default 1.0

        Returns
        -------
        str
            LaTex string
        """
        assert img.dim() == 3
        img_mask = torch.zeros_like(img, dtype=torch.long)  # squeeze channel
        hyps = self.bttr.beam_search(img.unsqueeze(0), img_mask, beam_size, max_len)
        best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** alpha))
        return vocab.indices2label(best_hyp.seq)

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        # Using the default logging method
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        hyps = self.bttr.beam_search(
            batch.imgs, batch.mask, self.hparams.beam_size, self.hparams.max_len
        )
        best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** self.hparams.alpha))

        self.exprate_recorder(best_hyp.seq, batch.indices[0])
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Batch, _):
        hyps = self.bttr.beam_search(
            batch.imgs, batch.mask, self.hparams.beam_size, self.hparams.max_len
        )
        best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** self.hparams.alpha))
        self.exprate_recorder(best_hyp.seq, batch.indices[0])

        self.test_outputs.append((batch.img_bases[0], vocab.indices2label(best_hyp.seq)))

        return batch.img_bases[0], vocab.indices2label(best_hyp.seq)

    # Changed for lightning 2.0+ compatibility
    def on_test_epoch_end(self) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"ExpRate: {exprate}")

        print(f"length of total file: {len(self.test_outputs)}")
        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_base, pred in self.test_outputs:
                content = f"%{img_base}\n${pred}$".encode()
                with zip_f.open(f"{img_base}.txt", "w") as f:
                    f.write(content)

        self.test_outputs.clear()

    # TODO: Add the configurations required for finetuning
    def configure_optimizers(self):
        
        # Apply different lr to finetuning (encoder) & groud-up training (decoder)
        encoder_params = []
        decoder_params = []

        for name, params in self.named_parameters():
            if "encoder" in name:
                encoder_params.append(params)
            elif "decoder" in name:
                decoder_params.append(params)

        # Test for the trainable params
        print("Params states check: ")
        for name, param in self.named_parameters():
            print(f"{name}: {param.requires_grad}")

        # Filter out the intially trainable encoder params (BatchNorm Layers)
        encoder_trainable_params = filter(lambda p: p.requires_grad, encoder_params)

        optimizer = optim.Adadelta([
            {'params': encoder_trainable_params, 'lr': self.hparams.finetune_learning_rate, 'eps': 1e-6, 'weight_decay': 1e-4},
            {'params': decoder_params, 'lr': self.hparams.learning_rate, 'eps': 1e-6, 'weight_decay': 1e-4}
        ])

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class GradualUnfreeze(BaseFinetuning):
    """
    Gradually unfreeze the encoder layers during training.
    For Finetuning use.
    """
    def __init__(self, unfreeze_n_epoch=2):
        super().__init__()
        self.unfreeze_n_epoch = unfreeze_n_epoch

    def freeze_before_training(self, pl_module):
        # This freeze method freeze the DenseNet except for the BatchNorm layers
        self.freeze(pl_module.bttr.encoder.model.features)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # Unfreeze the 3rd _dense, 2nd _transition, 2nd _dense sequentially
        # Remember to add dropout (n=0.2) while unfreezing
        if current_epoch % self.unfreeze_n_epoch == 0:
            i = current_epoch // self.unfreeze_n_epoch
            # DenseBlock3
            if i < 24:
                # Unfreeze
                self.unfreeze_and_add_param_group(pl_module.get_submodule(f'bttr.encoder.model.features.8.denselayer{24-i}.conv1')
                                                  ,optimizer=optimizer
                                                  ,lr=pl_module.hparams.finetune_learning_rate)
                self.unfreeze_and_add_param_group(pl_module.get_submodule(f'bttr.encoder.model.features.8.denselayer{24-i}.conv2')
                                                    ,optimizer=optimizer
                                                    ,lr=pl_module.hparams.finetune_learning_rate)
                # Add dropout
                pl_module.get_submodule(f'bttr.encoder.model.features.8.denselayer{24-i}').drop_rate = 0.2

            # TransitionBlock2
            elif i == 24:
                self.unfreeze_and_add_param_group(pl_module.get_submodule('bttr.encoder.model.features.7.conv')
                                                  ,optimizer=optimizer
                                                  ,lr=pl_module.hparams.finetune_learning_rate)
            
            # DenseBlock2
            elif i < 37:
                self.unfreeze_and_add_param_group(pl_module.get_submodule(f'bttr.encoder.model.features.6.denselayer{37-i}.conv1')
                                                  ,optimizer=optimizer
                                                  ,lr=pl_module.hparams.finetune_learning_rate)
                self.unfreeze_and_add_param_group(pl_module.get_submodule(f'bttr.encoder.model.features.6.denselayer{37-i}.conv2')
                                                    ,optimizer=optimizer
                                                    ,lr=pl_module.hparams.finetune_learning_rate)
                pl_module.get_submodule(f'bttr.encoder.model.features.6.denselayer{37-i}').drop_rate = 0.2