import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


# 이미 완성된 MultiDAE의 코드를 참고하여 그 아래 MultiVAE의 코드를 완성해보세요!
class MultiDAE(nn.Module):
    """
    Container module for Multi-DAE.

    Multi-DAE : Denoising Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, genre_emb=None, dropout=0.5):
        super(MultiDAE, self).__init__()

        self.device = torch.device("cuda")
        self.item_genre = torch.Tensor(genre_emb.values).to(self.device)

        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]  # 리스트를 역으로 뒤집음

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        self.drop = nn.Dropout(dropout)

        self.init_weights()

    def forward(self, input):
        # print('input.shape :', input.shape)
        h = F.normalize(input)
        h = self.drop(h)
        h = torch.cat((self.item_genre, h), 0)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.tanh(h)

        item_genre_emb, reconstructed_h = h.split([self.item_genre.shape[0], input.shape[0]], 0)
        return reconstructed_h

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, genre_emb=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.device = torch.device("cuda")
        self.item_genre = torch.Tensor(genre_emb.values).to(self.device)
        # self.item_title = torch.Tensor(title_emb.values).to(self.device)  # 추가

        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]

        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])

        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        self.input = input

        mu, logvar = self.encode(self.input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, input):
        h = F.normalize(input)
        # print('self.item_genre.device :', self.item_genre.device)
        # print('h.device :', h.device)
        # h = torch.cat((self.item_genre, self.item_title, h), 0)
        h = torch.cat((self.item_genre, h), 0)
        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:  # 이게 뭐지..?
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)

        # item_genre_emb, item_title_emb, reconstructed_h = h.split(
        # [self.item_genre.shape[0], self.item_title.shape[0], self.input.shape[0]], 0)
        item_genre_emb, reconstructed_h = h.split([self.item_genre.shape[0], self.input.shape[0]], 0)
        return reconstructed_h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


def loss_function_vae(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD


def loss_function_dae(recon_x, x):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    return BCE
