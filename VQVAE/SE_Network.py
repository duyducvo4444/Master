import torch
import torch.nn as nn
from VQVAE.EnDecoder import Encoder, Decoder
from VQVAE.VQ import VectorQuantizer


class SENetwork(nn.Module):
    def __init__(self, input_size, num_hiddens, batch_size, num_embeddings, embedding_dim, commitment_cost):
        super(SENetwork, self).__init__()
        self.encoder = Encoder(input_size=input_size, num_hiddens=num_hiddens)
        self.decoder = Decoder(input_size=input_size, num_hiddens=num_hiddens)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.batch_size = batch_size

    def forward(self, mag, phase):
        # xr and xi currently have size of [batch, features, sequence]
        z = self.encoder(mag)
        loss, quantized, perplexity, _ = self.vq(z)
        y_mag = self.decoder(quantized)
        return loss, y_mag, phase, perplexity
