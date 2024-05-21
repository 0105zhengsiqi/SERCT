# Code based on https://github.com/microsoft/unilm/blob/master/wavlm

import math
import torch
import torch.nn as nn
from typing import List, Tuple
from .cnn_utils import TransposeLast, Fp32LayerNorm, Fp32GroupNorm

class CNNEncoder(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int, int]] = [(512, 10, 5, 0)] + [(512, 3, 2, 0)] * 4 + [(512, 2, 2, 0)] + [(512, 2, 2, 1)],  # (dim, kernel_size, stride, padding)
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        conv_type: str = "default"
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            padding=0,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias, padding=padding)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(
                    make_conv(), 
                    nn.Dropout(p=dropout), 
                    nn.GELU()
                )

        self.conv_type = conv_type
        if self.conv_type == "default":
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 4, "invalid conv definition: " + str(cl)
                (dim, k, stride, padding) = cl

                self.conv_layers.append(
                    block(
                        in_d,
                        dim,
                        k,
                        stride,
                        padding,
                        is_layer_norm=(mode == "layer_norm"),
                        is_group_norm=(mode == "default" and i == 0),
                        conv_bias=conv_bias,
                    )
                )
                in_d = dim
        elif self.conv_type == "conv2d":
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl

                self.conv_layers.append(
                    torch.nn.Conv2d(in_d, dim, k, stride)
                )
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
        elif self.conv_type == "custom":
            in_d = 1
            idim = 80
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl
                self.conv_layers.append(
                    torch.nn.Conv2d(in_d, dim, k, stride, padding=1)
                )
                self.conv_layers.append(
                    torch.nn.LayerNorm([dim, idim])
                )
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
                if (i + 1) % 2 == 0:
                    self.conv_layers.append(
                        torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
                    )
                    idim = int(math.ceil(idim / 2))
        else:
            raise ValueError(f'{self.conv_type} not supported.')

    def forward(self, x):
        # B x T -> B x C x T
        x = x.unsqueeze(1)
        if self.conv_type == "custom":
            for conv in self.conv_layers:
                if isinstance(conv, nn.LayerNorm):
                    x = x.transpose(1, 2)
                    x = conv(x).transpose(1, 2)
                else:
                    x = conv(x)
            x = x.transpose(2, 3).contiguous()
            x = x.view(x.size(0), -1, x.size(-1))
        else:
            for conv in self.conv_layers:
                x = conv(x)
            if self.conv_type == "conv2d":
                b, c, t, f = x.size()
                x = x.transpose(2, 3).contiguous().view(b, c * f, t)
        return x
