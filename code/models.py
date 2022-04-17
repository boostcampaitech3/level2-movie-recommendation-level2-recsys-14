import math
from copy import deepcopy

import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import Linear
from torch.nn import functional as F
from torch.nn.functional import normalize

from utils.distributions import *
from utils.nn import normal_init, he_init, NonLinear


def swish(x):
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


def loss_function_vae(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD


def loss_function_dae(recon_x, x):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    return BCE


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args

    # AUXILIARY METHODS
    def add_pseudoinputs(self):

        nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)

        self.means = NonLinear(self.args.number_components, np.prod(self.args.input_size), bias=False,
                               activation=nonlinearity)

        # init pseudo-inputs
        if self.args.use_training_data_init:
            self.means.linear.weight.data = self.args.pseudoinputs_mean
        else:
            normal_init(self.means.linear, self.args.pseudoinputs_mean, self.args.pseudoinputs_std)

        # create an idle input for calling pseudo-inputs
        self.idle_input = Variable(torch.eye(self.args.number_components, self.args.number_components),
                                   requires_grad=False)
        if self.args.cuda:
            self.idle_input = self.idle_input.cuda()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            if self.args.cuda:
                eps = torch.cuda.FloatTensor(std.size()).normal_()
            else:
                eps = torch.FloatTensor(std.size()).normal_()
            eps = Variable(eps)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def calculate_loss(self):
        return 0.

    def calculate_likelihood(self):
        return 0.

    def calculate_lower_bound(self):
        return 0.

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        return 0.


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=None):
        super(CompositePrior, self).__init__()

        if mixture_weights is None:
            mixture_weights = [3 / 20, 3 / 4, 1 / 10]
        self.mixture_weights = mixture_weights

        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)

        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)

        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)

        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)

    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)

        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)

        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]

        density_per_gaussian = torch.stack(gaussians, dim=-1)

        return torch.logsumexp(density_per_gaussian, dim=-1)


class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]

        x = F.dropout(x, p=dropout_rate, training=self.training)

        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)


"""
RecVAE
"""

class RecVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim):
        super(RecVAE, self).__init__()

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, user_ratings, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)

        if calculate_loss:
            if gamma:
                norm = user_ratings.sum(dim=-1)
                kl_weight = gamma * norm
            elif beta:
                kl_weight = beta

            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
            negative_elbo = -(mll - kld)

            return (mll, kld), negative_elbo

        else:
            return x_pred

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))

"""
Multi Denoising AutoEncoder
"""

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
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                     d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        self.drop = nn.Dropout(dropout)

        self.init_weights()

    def forward(self, input):
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


"""
Multi Variational AutoEncoder
"""

class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, genre_emb=None, title_emb=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.device = torch.device("cuda")
        self.item_genre = torch.Tensor(genre_emb.values).to(self.device)

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
        h = self.drop(h)
        h = torch.cat((self.item_genre, h), 0)

        mu, logvar = None, None
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

        item_genre_emb, reconstructed_h = h.split(
            [self.item_genre.shape[0], self.input.shape[0]], 0)
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


"""
H+vamp with 1 hidden layer in EVCF
"""

class Hvamp(Model):
    def __init__(self, args):
        super(Hvamp, self).__init__(args)

        self.args = args

        # encoder: q(z2 | x)
        self.q_z2_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            NonLinear(np.prod(self.args.input_size), self.args.hidden_size, gated=self.args.gated,
                      activation=nn.Tanh()),
        )

        self.q_z2_mean = Linear(self.args.hidden_size, self.args.z2_size)
        self.q_z2_logvar = NonLinear(self.args.hidden_size, self.args.z2_size,
                                     activation=nn.Hardtanh(min_val=-12., max_val=4.))

        # encoder: q(z1 | x, z2)
        self.q_z1_layers_x = nn.Sequential(
            nn.Dropout(p=0.5),
        )
        self.q_z1_layers_z2 = nn.Sequential(
        )
        self.q_z1_layers_joint = nn.Sequential(
            NonLinear(np.prod(self.args.input_size) + self.args.z2_size, self.args.hidden_size, gated=self.args.gated,
                      activation=nn.Tanh())
        )

        self.q_z1_mean = Linear(self.args.hidden_size, self.args.z1_size)
        self.q_z1_logvar = NonLinear(self.args.hidden_size, self.args.z1_size,
                                     activation=nn.Hardtanh(min_val=-12., max_val=4.))

        # decoder: p(z1 | z2)
        self.p_z1_layers = nn.Sequential(
            NonLinear(self.args.z2_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()),
        )

        self.p_z1_mean = Linear(self.args.hidden_size, self.args.z1_size)
        self.p_z1_logvar = NonLinear(self.args.hidden_size, self.args.z1_size,
                                     activation=nn.Hardtanh(min_val=-12., max_val=4.))

        # decoder: p(x | z1, z2)
        self.p_x_layers_z1 = nn.Sequential(
        )
        self.p_x_layers_z2 = nn.Sequential(
        )
        self.p_x_layers_joint = nn.Sequential(
            NonLinear(self.args.z1_size + self.args.z2_size, self.args.hidden_size, gated=self.args.gated,
                      activation=nn.Tanh())
        )

        if self.args.input_type == 'binary':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())
        if self.args.input_type == 'multinomial':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=None)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())
            self.p_x_logvar = NonLinear(self.args.hidden_size, np.prod(self.args.input_size),
                                        activation=nn.Hardtanh(min_val=-4.5, max_val=0))

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

        # add pseudo-inputs for VampPrior
        self.add_pseudoinputs()


    # AUXILIARY METHODS
    def calculate_loss(self, x, beta=1., average=False):
        '''
        :param x: input image(s)
        :param beta: a hyperparam for warmup
        :param average: whether to average loss or not
        :return: value of a loss function
        '''
        # pass through VAE
        x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, \
        z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = self.forward(x)

        # RE
        if self.args.input_type == 'binary':
            RE = log_Bernoulli(x, x_mean, dim=1)
        elif self.args.input_type == 'multinomial':
            RE = log_Softmax(x, x_mean, dim=1)  # ! Actually not Reconstruction Error but Log-Likelihood
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            RE = -log_Logistic_256(x, x_mean, x_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')

        # KL
        log_p_z1 = log_Normal_diag(z1_q, z1_p_mean, z1_p_logvar, dim=1)
        log_q_z1 = log_Normal_diag(z1_q, z1_q_mean, z1_q_logvar, dim=1)
        log_p_z2 = self.log_p_z2(z2_q)
        log_q_z2 = log_Normal_diag(z2_q, z2_q_mean, z2_q_logvar, dim=1)
        KL = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)

        loss = -RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL

    # ADDITIONAL METHODS
    def reconstruct_x(self, x):
        x_mean, _, _, _, _, _, _, _, _, _ = self.forward(x)
        return x_mean

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z2(self, x):
        x = self.q_z2_layers(x)

        z2_q_mean = self.q_z2_mean(x)
        z2_q_logvar = self.q_z2_logvar(x)
        return z2_q_mean, z2_q_logvar

    def q_z1(self, x, z2):
        x = self.q_z1_layers_x(x)

        z2 = self.q_z1_layers_z2(z2)

        h = torch.cat((x, z2), 1)

        h = self.q_z1_layers_joint(h)

        z1_q_mean = self.q_z1_mean(h)
        z1_q_logvar = self.q_z1_logvar(h)
        return z1_q_mean, z1_q_logvar

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_z1(self, z2):
        z2 = self.p_z1_layers(z2)

        z1_mean = self.p_z1_mean(z2)
        z1_logvar = self.p_z1_logvar(z2)
        return z1_mean, z1_logvar

    def p_x(self, z1, z2):
        z1 = self.p_x_layers_z1(z1)

        z2 = self.p_x_layers_z2(z2)

        h = torch.cat((z1, z2), 1)

        h = self.p_x_layers_joint(h)

        x_mean = self.p_x_mean(h)
        if self.args.input_type == 'binary' or self.args.input_type == 'multinomial':
            x_logvar = 0.
        else:
            x_mean = torch.clamp(x_mean, min=0. + 1. / 512., max=1. - 1. / 512.)
            x_logvar = self.p_x_logvar(h)
        return x_mean, x_logvar

    # the prior
    def log_p_z2(self, z2):
        # vamp prior
        # z2 - MB x M
        C = self.args.number_components

        # calculate params
        X = self.means(self.idle_input)

        # calculate params for given data
        z2_p_mean, z2_p_logvar = self.q_z2(X)  # C x M

        # expand z
        z_expand = z2.unsqueeze(1)
        means = z2_p_mean.unsqueeze(0)
        logvars = z2_p_logvar.unsqueeze(0)

        a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
        a_max, _ = torch.max(a, 1)  # MB
        # calculte log-sum-exp
        log_prior = (a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1)))  # MB

        return log_prior

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        # input normalization
        x = normalize(x, dim=1)

        # z2 ~ q(z2 | x)
        z2_q_mean, z2_q_logvar = self.q_z2(x)
        z2_q = self.reparameterize(z2_q_mean, z2_q_logvar)

        # z1 ~ q(z1 | x, z2)
        z1_q_mean, z1_q_logvar = self.q_z1(x, z2_q)
        z1_q = self.reparameterize(z1_q_mean, z1_q_logvar)

        # p(z1 | z2)
        z1_p_mean, z1_p_logvar = self.p_z1(z2_q)

        # x_mean = p(x|z1,z2)
        x_mean, x_logvar = self.p_x(z1_q, z2_q)

        return x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar


"""
EASE
"""


class EASE:
    """
    Embarrassingly Shallow Autoencoders model class
    """

    def __init__(self, lambda_):
        self.B = None
        self.lambda_ = lambda_

    def train(self, rating_matrix):
        """
        train pass
        :param rating_matrix: rating matrix
        """
        G = rating_matrix.T @ rating_matrix
        diag = list(range(G.shape[0]))
        G[diag, diag] += self.lambda_
        P = np.linalg.inv(G)

        # B = P * (X^T * X − diagMat(γ))
        self.B = P / -np.diag(P)
        min_dim = min(*self.B.shape)
        self.B[range(min_dim), range(min_dim)] = 0

    def forward(self, user_row):
        """
        forward pass
        """
        return user_row @ self.B
