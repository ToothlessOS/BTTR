import math
from typing import Tuple

import pytorch_lightning as pl
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from torch import FloatTensor, LongTensor

from .pos_enc import ImageRotaryEmbed, ImgPosEnc

class Encoder(pl.LightningModule):
    def __init__(self, d_model: int, growth_rate: int, num_layers: int, is_pretrained=True):
        super().__init__()
        
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        # Freeze all but the last denseblock (for fine-tuning)
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.features.denseblock3.parameters():
            param.requires_grad = True

        # First, we want to intialize the parameters in denseblock3
        # Reinitialize specified block
        for name, module in self.model.features.denseblock3.named_children():
            # Conv layers
            nn.init.kaiming_normal_(module.conv1.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(module.conv2.weight, mode='fan_out', nonlinearity='relu')
            # Batch norm layers
            nn.init.ones_(module.norm1.weight)
            nn.init.ones_(module.norm2.weight)
            nn.init.zeros_(module.norm1.bias)
            nn.init.zeros_(module.norm2.bias)

            # Then, we add the dropout layers(n=0.2) to every _DenseLayer in denseblock3
            module.drop_date = 0.2

        # Remove the Transition Layer 3, Dense Block 4 and Classification Layer from the model
        self.model.features = nn.Sequential(
            *list(self.model.features.children())[:-3]  # Keep layers up to Dense Block 3
        )

        self.model.classifier = nn.Identity()

        self.feature_proj = nn.Sequential(
            nn.Conv2d(1024, d_model, kernel_size=1), # self.model.out_channels
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(d_model)

        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)

    def forward(
        self, img: FloatTensor, img_mask: LongTensor
    ) -> Tuple[FloatTensor, LongTensor]:
        """encode image to feature

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']

        Returns
        -------
        Tuple[FloatTensor, LongTensor]
            [b, t, d], [b, t]
        """
        # Img channel duplication: [b, 1, h', w'] -> [b, 3, h', w']
        # This is necessary for the provided greyscale dataset
        # img: torch.Size([1, 3, 54, 53])
        # img_mask: torch.Size([1, 54, 53])
        img = img.expand(-1, 3, -1, -1)

        # extract feature
        feature = self.model.features(img)
        feature = self.feature_proj(feature)

        # Functions for downsampling the mask
        # Designed accordingly with the downsampling layers of DenseNet-121
        # 1: 7 * 7 conv, stride 2
        img_mask =  _downsample_mask_conv(img_mask, kernel_size=7, stride=2, padding=3)
        # 2: 3 * 3 max pool, stride 2, padding 1
        img_mask = _downsample_mask_max_pool(img_mask, kernel_size=3, stride=2, padding=1)
        # 3: 2 * 2 avg pool, stride 2
        img_mask = _downsample_mask_avg_pool(img_mask, kernel_size=2, stride=2)
        # 4：2 * 2 avg pool, stride 2
        mask = _downsample_mask_avg_pool(img_mask, kernel_size=2, stride=2)

        # proj
        feature = rearrange(feature, "b d h w -> b h w d")
        feature = self.norm(feature)

        # positional encoding
        feature = self.pos_enc_2d(feature, mask)

        # flat to 1-D
        feature = rearrange(feature, "b h w d -> b (h w) d")
        mask = rearrange(mask, "b h w -> b (h w)")
        return feature, mask
    
def _downsample_mask_avg_pool(mask: LongTensor, kernel_size: int, stride: int) -> LongTensor:
    return F.avg_pool2d(mask.float(), 
                kernel_size=kernel_size, 
                stride=stride).bool()

def _downsample_mask_max_pool(mask: LongTensor, kernel_size: int, stride: int, padding: int) -> LongTensor:
    return F.max_pool2d(mask.float(), 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding).bool()

def _downsample_mask_conv(mask: LongTensor, kernel_size: int, stride: int, padding: int) -> LongTensor:

    # Pad to match conv's implicit padding (3 on all sides)
    mask_padded = F.pad(mask.float(), (padding, padding, padding, padding), mode='constant', value=0)
    
    # Apply max pooling with matching kernel/stride
    return F.max_pool2d(mask_padded, 
                    kernel_size=kernel_size, 
                    stride=stride).bool()