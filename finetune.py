"""
Use this script when using the pretrained DenseNet-121 Model
"""

from pytorch_lightning.cli import LightningCLI

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr_finetune import LitBTTR

if __name__ == "__main__":
    cli = LightningCLI(LitBTTR, CROHMEDatamodule)
