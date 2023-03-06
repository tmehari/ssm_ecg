# adapted from https://github.com/HazyResearch/state-spaces/blob/main/example.py

import torch
import torch.nn as nn
import sys
import os
egg_path = "/home/mehari/anaconda3/envs/stsp/lib/python3.9/cauchy_mult-0.0.0-py3.9-linux-x86_64.egg"
egg_path = "/home/mehari/clones/stsp/code/extensions/cauchy/cauchy_mult-0.0.0-py3.9-linux-x86_64.egg"
egg_path = os.path.join(os.getcwd(),  "code/extensions/cauchy/cauchy_mult-0.0.0-py3.9-linux-x86_64.egg")
sys.path.append(egg_path)

import cauchy_mult
from dl_models.s4 import S4
from .basic_conv1d import bn_drop_lin

class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output,
        d_model=512,
        d_state=8,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        l_max=1024,
        transposed_input=True,
        bn = False,
        bidirectional=False,
        use_meta_information_in_head=False
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.transposed_input = transposed_input
        self.encoder = nn.Conv1d(
            d_input, d_model, 1) if transposed_input else nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(
                    d_model=d_model,
                    l_max=l_max,
                    bidirectional=bidirectional,
                    postact='glu',
                    dropout=dropout,
                    transposed=True,
                    d_state=d_state
                )
            )
            if bn:
                self.norms.append(nn.BatchNorm1d(d_model))
            else:
                self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))

        # Linear decoder
        # MODIFIED TO ALLOW FOR MODELS WITHOUT DECODER
        if(d_output is None):
            self.decoder = None
        else:
            self.decoder = nn.Linear(d_model + 64 if use_meta_information_in_head else d_model, d_output)
        if use_meta_information_in_head:
            meta_modules = bn_drop_lin(7, 64, bn=False,actn=nn.ReLU()) +\
            bn_drop_lin(64, 64, bn=True, p=0.5, actn=nn.ReLU()) + bn_drop_lin(64, 64, bn=True, p=0.5, actn=nn.ReLU())
            self.meta_head = nn.Sequential(*meta_modules)

        
    def forward(self, x, rate=1.0):
        """
        Input x is shape (B, d_input, L) if transposed_input else (B, L, d_input)
        """
        x = self.encoder(
            x)  # (B, d_input, L) -> (B, d_model, L) if transposed_input else (B, L, d_input) -> (B, L, d_model)
        
        if(self.transposed_input is False):
            x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            
            # Apply S4 block: we ignore the state input and output
            
            z, _ = layer(z, rate=rate)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        if self.decoder is not None:
            x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x
    
    def forward_with_meta(self, x, meta_feats, rate=1.0):
        """
        Input x is shape (B, d_input, L) if transposed_input else (B, L, d_input)
        """
        x = self.encoder(
            x)  # (B, d_input, L) -> (B, d_model, L) if transposed_input else (B, L, d_input) -> (B, L, d_model)
        
        if(self.transposed_input is False):
            x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            
            # Apply S4 block: we ignore the state input and output
            
            z, _ = layer(z, rate=rate)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)
        meta_feats = self.meta_head(meta_feats)
        x = torch.cat([x, meta_feats], axis=1)
        
        # Decode the outputs
        if self.decoder is not None:
            x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x
