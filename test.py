from pytorch_lightning import Trainer

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR
from bttr.lit_bttr import VisualizeAndValidateCallback

test_year = "2014"
# ckp_path = "lightning_logs/version_0/checkpoints/epoch=259-step=97759.ckpt"
ckp_path = "lightning_logs/Attempt0Part1/checkpoints/epoch=27-step=42028-val_ExpRate=0.2162.ckpt"

if __name__ == "__main__":
    trainer = Trainer(logger=False,
                        num_nodes=1,
                        devices=[0],
                        callbacks=[VisualizeAndValidateCallback()])

    dm = CROHMEDatamodule(test_year=test_year)

    model = LitBTTR.load_from_checkpoint(ckp_path)

    trainer.test(model, datamodule=dm)
