import torch.nn as nn
from .transformer_encoder import TransformerEncoder
from .cnn_encoder import CNNEncoder
from typing import List, Tuple


class CNNTransformer(nn.Module):
    
    def __init__(
        self,
        # args for cnn
        cnn_mode: str = 'default',
        conv_layers: List[Tuple[int, int, int, int]] = [(512, 10, 5, 0)] + [(512, 3, 2, 0)] * 4 + [(512, 2, 2, 0)] + [(512, 2, 2, 1)],  # (dim, kernel_size, stride, padding)
        cnn_dropout: float = 0.0,
        conv_bias: bool = False,
        conv_type: str = "default",
        # args for transformer
        input_dim: int = 1024,
        length: int = 326,
        ffn_embed_dim: int = 512, 
        num_layers: int = 4, 
        num_heads: int = 8, 
        num_classes: int = 7, 
        trans_dropout: float = 0.1,
        bias: bool = True,
        activation: str = 'relu'
    ) -> None:
        
        super().__init__()
        self.cnn_mode = cnn_mode
        self.conv_layers = conv_layers
        self.cnn_dropout = cnn_dropout
        self.conv_bias = conv_bias
        self.conv_type = conv_type

        self.input_dim = input_dim
        self.length = length
        self.ffn_embed_dim = ffn_embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.trans_dropout = trans_dropout
        self.bias = bias
        self.activation = activation

        self.cnn_encoder = CNNEncoder(
            conv_layers=self.conv_layers,
            dropout=self.cnn_dropout,
            mode=self.cnn_mode,
            conv_bias=self.conv_bias,
            conv_type=self.conv_type
        )
        self.transformer_encoder = TransformerEncoder(
            input_dim=self.input_dim,
            length=self.length,
            ffn_embed_dim=self.ffn_embed_dim, 
            num_layers=self.num_layers, 
            num_heads=self.num_heads, 
            num_classes=self.num_classes, 
            dropout=self.trans_dropout,
            bias=self.bias,
            activation=self.activation
        )
        self.cnn_out_dim = self.conv_layers[-1][0]
        self.cnn_transformer_proj = (
            nn.Linear(self.cnn_out_dim, self.input_dim)
            if self.cnn_out_dim != self.input_dim
            else None
        )
        self.layer_norm = nn.LayerNorm(self.cnn_out_dim)
        
    def forward(self, x):
        x = self.cnn_encoder(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        if self.cnn_transformer_proj is not None:
            x = self.cnn_transformer_proj(x)
        pred = self.transformer_encoder(x)
        return pred
