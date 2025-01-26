from pytorch_lightning.cli import LightningCLI

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

cli = LightningCLI(LitBTTR, CROHMEDatamodule)
