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
        
        if is_pretrained:
            self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            self.model = models.densenet121(weights=None)

        # Freeze all but the last denseblock (for fine-tuning)
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.features.denseblock3.parameters():
            param.requires_grad = True

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
        # 2: 3 * 3 max pool, stride 2
        img_mask = _downsample_mask_max_pool(img_mask, kernel_size=3, stride=2)
        # 3: 2 * 2 avg pool, stride 2
        img_mask = _downsample_mask_avg_pool(img_mask, kernel_size=2, stride=2)
        # 4：2 * 2 avg pool, stride 2
        mask = _downsample_mask_avg_pool(img_mask, kernel_size=2, stride=2)

        print(mask)

        # For testing, we check the shape
        # TODO: Dimension mismatch
        print(feature.shape) # [1, 1024, 3, 3]
        print(mask.shape) # [1, 4, 4]

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

def _downsample_mask_max_pool(mask: LongTensor, kernel_size: int, stride: int) -> LongTensor:
    return F.max_pool2d(mask.float(), 
            kernel_size=kernel_size, 
            stride=stride,
            ceil_mode=True).bool()

def _downsample_mask_conv(mask: LongTensor, kernel_size: int, stride: int, padding: int) -> LongTensor:

    # Pad to match conv's implicit padding (3 on all sides)
    mask_padded = F.pad(mask.float(), (padding, padding, padding, padding), mode='constant', value=0)
    
    # Apply max pooling with matching kernel/stride
    return F.max_pool2d(mask_padded.float(), 
                    kernel_size=kernel_size, 
                    stride=stride).bool()