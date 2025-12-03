import torch
import torch.nn as nn

class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=4, num_heads=4):
        super().__init__()

        self.input_linear = nn.Linear(in_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.out_dim = hidden_dim

    def forward(self, x, attn_bias=None):
        h = self.input_linear(x)

        if attn_bias is not None:
            # Add bias to attention scores
            self.transformer.layers[0].self_attn.bias = attn_bias

        h = self.transformer(h)  # shape: [n, hidden]
        return h
