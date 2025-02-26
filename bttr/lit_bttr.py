import zipfile

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor

from bttr.datamodule import Batch, vocab
from bttr.model.bttr import BTTR
from bttr.utils import ExpRateRecorder, Hypothesis, ce_loss, to_bi_tgt_out

from pytorch_lightning.callbacks import Callback

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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

        # Record the encoded features during test-time
        extracted_features = self.bttr.encoder.extracted_features

        return batch.img_bases[0], vocab.indices2label(best_hyp.seq), extracted_features

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

    def configure_optimizers(self):
        optimizer = optim.Adadelta(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=1e-6,
            weight_decay=1e-4,
        )

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
    
# Callback during testing for visualization & validation (featuer extraction)
class VisualizeAndValidateCallback(Callback):
    def on_test_epoch_start(self, trainer, pl_module):
        print("Testing started")
        # Extract and visualiza the word embeddings + Apply TSNE
        embedding = pl_module.get_submodule("bttr.decoder.word_embed.0").weight.data.cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(embedding)

        # Visualization
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(features_tsne[:, 0], features_tsne[:, 1])
        plt.title('TSNE Visualization of BTTR Model Output Features in 2D')
        plt.xlabel('TSNE Dimension 1')
        plt.ylabel('TSNE Dimension 2')
        plt.savefig("lightning_logs/FeatureExtraction/PretrainedDenseNet+dropout/features/word_embed.png")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # The features extracted
        filename = outputs[0]
        encoded_features = outputs[2].squeeze(0).cpu().numpy()
        
        # Apply TSNE on the extracted features
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        features_tsne = tsne.fit_transform(encoded_features)

        # Visualization
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(features_tsne[:, 0], features_tsne[:, 1])
        plt.title(f'TSNE Visualization of Encoded Features of {filename} in 2D')
        plt.xlabel('TSNE Dimension 1')
        plt.ylabel('TSNE Dimension 2')
        plt.savefig(f"lightning_logs/FeatureExtraction/PretrainedDenseNet+dropout/features/{filename}.png")
