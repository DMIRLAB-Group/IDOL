"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributions as D
from torch.nn import functional as F
from .components.beta import BetaVAE_MLP
from .components.transition import (NPChangeInstantaneousTransitionPrior, NPInstantaneousTransitionPrior)
from .components.mlp import MLPEncoder, MLPDecoder, Inference
from .metrics.correlation import compute_mcc
import matplotlib.pyplot as plt

import wandb

def plot_sparsity_matrix(alphas, title):
    mat_a = torch.tensor(alphas).numpy()
    mat_a_values = np.round(mat_a, decimals=2)
    fig, ax = plt.subplots()
    fig_alpha = ax.imshow(mat_a, cmap='Greens', vmin=0, vmax=1)
    for i in range(mat_a.shape[0]):
        for j in range(mat_a.shape[1]):
            ax.text(j, i, mat_a_values[i, j],
                    ha='center', va='center', color='black')
    cbar = fig.colorbar(fig_alpha)

    ax.set_xticks(np.arange(mat_a.shape[1]))
    ax.set_yticks(np.arange(mat_a.shape[0]))

    ax.set_xticklabels(np.arange(mat_a.shape[1])+1)
    ax.set_yticklabels(np.arange(mat_a.shape[0])+1)

    cbar.set_label('Alpha')

    ax.set_title(title)
    ax.set_xlabel('From')
    ax.set_ylabel('To')

    return fig


class InstantaneousProcess(pl.LightningModule):
    def __init__(
        self, 
        input_dim,
        z_dim,
        z_dim_fix, 
        z_dim_change,
        lag,
        nclass,
        hidden_dim=128,
        embedding_dim=8,
        lr=1e-4,
        beta=0.0025,
        gamma=0.0075,
        theta=0.2,
        decoder_dist='gaussian',
        correlation='Pearson',
        enable_flexible_sparsity=False,
        w_hist=None,
        w_inst=None):
        '''Nonlinear ICA for time-varing causal processes with instantaneous causal effects'''
        super().__init__()
        self.z_dim = z_dim
        self.z_dim_fix = z_dim_fix
        self.z_dim_change = z_dim_change
        assert (self.z_dim == self.z_dim_fix + self.z_dim_change)
        self.lag = lag
        self.input_dim = input_dim
        self.lr = lr
        self.lag = lag
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.nclass = nclass

        if enable_flexible_sparsity:
            self.w_hist = w_hist
            self.w_inst = w_inst
        else:
            self.w_hist = [1.0] * z_dim
            self.w_inst = [1.0] * z_dim

        # Domain embeddings (dynamics)
        self.embed_func = nn.Embedding(nclass, embedding_dim)
        # Recurrent/Factorized inference
        self.net = BetaVAE_MLP(input_dim=input_dim, 
                                z_dim=z_dim, 
                                hidden_dim=hidden_dim)

        self.transition_prior_fix = NPInstantaneousTransitionPrior(lags=lag, 
                                                      latent_size=z_dim, 
                                                      num_layers=3, 
                                                      hidden_dim=hidden_dim)
        self.transition_prior_change = NPChangeInstantaneousTransitionPrior(lags=lag, 
                                                        latent_size=z_dim,
                                                        embedding_dim=embedding_dim,
                                                        num_layers=3, 
                                                        hidden_dim=hidden_dim)
        
                                                            
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))

    @property
    def base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)
    
    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def reconstruction_loss(self, x, x_recon, distribution):
        batch_size = x.size(0)
        assert batch_size != 0

        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(
                x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'gaussian':
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'sigmoid_gaussian':
            x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

        return recon_loss

    def loss_function(self, x, x_recon, mus, logvars, zs, embeddings):
        '''
        VAE ELBO loss: recon_loss + kld_loss (past: N(0,1), future: N(0,1) after flow) + sparsity_loss
        '''
        batch_size, length, _ = x.shape

        # Sparsity loss
        sparsity_loss = 0
        # fix
        # if self.z_dim_fix>0:
        #     unmasked_alphas_fix = self.alphas_fix()
        #     mask_fix = (unmasked_alphas_fix > 0.1).float()
        #     alphas_fix = unmasked_alphas_fix * mask_fix

        # Recon loss
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag]), torch.ones_like(logvars[:,:self.lag]))
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag]),dim=-1),dim=-1)
        log_qz_normal = torch.sum(torch.sum(log_qz[:,:self.lag],dim=-1),dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()

        # Future KLD
        kld_future = []
        # fix
        if self.z_dim_fix>0:
            log_qz_laplace = log_qz[:,self.lag:,:self.z_dim_fix]
            residuals, logabsdet, hist_jac = self.transition_prior_fix.forward(zs[:,:,:self.z_dim_fix])
            log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + logabsdet
            kld_laplace = (torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace) / (length-self.lag)
            kld_future.append(kld_laplace)
            # sparsity on history
            # sparsity_loss = F.l1_loss(hist_jac, torch.zeros_like(hist_jac))

            # import pdb
            # pdb.set_trace()

            trans_show = []
            inst_show = []
            numm = 0
            p_list = [2, 7, 11]
            p2_list = [0.0, 1.0, 4.0]
            cnt = 0


            for jac in hist_jac:
                sparsity_loss = sparsity_loss + self.w_inst[cnt] * F.l1_loss(jac[:,0,self.lag*self.z_dim_fix:], torch.zeros_like(jac[:,0,self.lag*self.z_dim_fix:]), reduction='sum')
                sparsity_loss = sparsity_loss + self.w_hist[cnt] * F.l1_loss(jac[:,0,:self.lag*self.z_dim_fix], torch.zeros_like(jac[:,0,:self.lag*self.z_dim_fix]), reduction='sum')
                cnt += 1
                numm = numm + jac.numel()
                trans_show.append(jac[:,0,:self.lag*self.z_dim_fix].detach().cpu())
                inst_cur = jac[:,0,self.lag*self.z_dim_fix:].detach().cpu()
                inst_cur = torch.nn.functional.pad(inst_cur, (0,self.z_dim_fix-inst_cur.shape[1],0,0), mode='constant', value=0)
                inst_show.append(inst_cur)
            trans_show = torch.stack(trans_show, dim=1).abs().mean(dim=0)
            inst_show = torch.stack(inst_show, dim=1).abs().mean(dim=0)
            sparsity_loss = sparsity_loss / numm


            self.trans_show = trans_show
            self.inst_show = inst_show
        # change
        if self.z_dim_change>0:
            assert(0)
            # log_qz_laplace = log_qz[:,self.lag:,self.z_dim_fix:]
            # residuals, logabsdet = self.transition_prior_change.forward(zs[:,:,self.z_dim_fix:], embeddings, alphas_change)
            # log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + logabsdet
            # kld_laplace = (torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace) / (length-self.lag)
            # kld_future.append(kld_laplace)
        kld_future = torch.cat(kld_future, dim=-1)
        kld_future = kld_future.mean()

        return sparsity_loss ,recon_loss, kld_normal, kld_future
    
    def forward(self, batch):
        # Prepare data
        if self.z_dim_change>0:
            x, y, c = batch['xt'], batch['yt'], batch['ct']
            c = torch.squeeze(c).to(torch.int64)
            embeddings = self.embed_func(c)
        else:
            x, y = batch['xt'], batch['yt']
            embeddings = None
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        

        # Inference & Reshape to time-series format
        x_recon, mus, logvars, zs = self.net(x_flat)
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)

        return x, x_recon, mus, logvars, zs, embeddings

    def training_step(self, batch, batch_idx):
        x, x_recon, mus, logvars, zs, embeddings = self.forward(batch)
        sparsity_loss, recon_loss, kld_normal, kld_future = self.loss_function(x, x_recon, mus, logvars, zs, embeddings)

        # VAE training
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_future + self.theta * sparsity_loss
        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_normal", self.beta * kld_normal)
        self.log("train_kld_future", self.gamma * kld_future)
        self.log("train_sparsity_loss", self.theta * sparsity_loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, x_recon, mus, logvars, zs, embeddings = self.forward(batch)
        sparsity_loss, recon_loss, kld_normal, kld_future = self.loss_function(x, x_recon, mus, logvars, zs, embeddings)

        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()
        zt_true = batch["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        # Draw sparsity matrix
        # import pdb
        # pdb.set_trace()
        # softmax through dim 1 on numpy
        # hist_jac_soft = torch.softmax(torch.abs(self.hist_jac), dim=1)
        # hist_jac_fig = plot_sparsity_matrix(hist_jac_soft, "Sparsity Matrix (hist_jac_fig)")
        # self.logger.experiment.log({'alpha_softmax': wandb.Image(hist_jac_fig)})

        trans_show_dir = torch.abs(self.trans_show)
        # trans_show_fig = plot_sparsity_matrix(trans_show_dir, "Sparsity Matrix (hist_jac_fig)")
        trans_show_fig = plot_sparsity_matrix(trans_show_dir, "Transition Matrix")
        self.logger.experiment.log({'trans': wandb.Image(trans_show_fig)})

        inst_show_dir = torch.abs(self.inst_show)
        # inst_show_fig = plot_sparsity_matrix(inst_show_dir, "Sparsity Matrix (inst_jac_fig)")
        inst_show_fig = plot_sparsity_matrix(inst_show_dir, "Instantaneous Matrix")
        self.logger.experiment.log({'inst': wandb.Image(inst_show_fig)})

        # if self.z_dim_fix>0:
        #     fig_fix = plot_sparsity_matrix(self.alphas_fix().detach().cpu(), "Sparsity Matrix (Fix)")
        #     self.logger.experiment.log({'alpha': wandb.Image(fig_fix)})
        # if self.z_dim_change>0:
        #     fig_change = plot_sparsity_matrix(self.alphas_change().detach().cpu(), "Sparsity Matrix (Change)")
        #     self.logger.experiment.log({'alpha': wandb.Image(fig_change)})

        # Validation loss
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_future + self.theta * sparsity_loss
        self.log("val_mcc", mcc) 
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_normal", self.beta * kld_normal)
        self.log("val_kld_future", self.gamma * kld_future)
        self.log("val_sparsity_loss", self.theta * sparsity_loss)

        return loss

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return [opt_v], []
