#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from timm import create_model
import warnings


# ---------------------- Basic Utility Layers ----------------------

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


def reparametrize_trick(mu, sigma):
    sigma = sigma.sqrt()
    eps = torch.randn_like(sigma)
    return mu + eps * sigma


# ---------------------- Group Generalized Mean Pooling ----------------------

class GGeM(nn.Module):
    """
    Group Generalized Mean Pooling (GGeM)
    Splits embedding channels into groups and learns one p per group.
    """
    def __init__(self, num_channels=768, num_groups=8, eps=1e-6, init_p=3.0):
        super(GGeM, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps

        assert num_channels % num_groups == 0, \
            f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"

        # One learnable p per group
        self.p = nn.Parameter(torch.ones(num_groups) * init_p)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, D] — token embeddings from ViT
        Returns:
            Tensor of shape [B, D] — pooled embeddings
        """
        B, N, D = x.shape
        Dg = D // self.num_groups

        # reshape [B, N, D] -> [B, N, G, Dg]
        x = x.view(B, N, self.num_groups, Dg)

        # group-wise p
        p = torch.clamp(self.p, min=1e-6).view(1, 1, self.num_groups, 1)

        # group-wise GeM pooling over tokens
        x = (x.clamp(min=self.eps).pow(p)).mean(dim=1).pow(1.0 / p)  # [B, G, Dg]

        # flatten back to [B, D]
        return x.view(B, -1)


# ---------------------- Main ViT Model ----------------------

class Model(nn.Module):
    def __init__(self, mu_dim=768, sigma_dim=1, sigma_init=1e-3,
                 dropout_rate=0.1, setting='btl', model_name='vit_base_patch16_224.dino',
                 num_groups=8):
        """
        Vision Transformer backbone with uncertainty-aware heads.
        Compatible with settings: 'btl', 'dul', 'mcd', 'triplet'
        """
        super().__init__()
        self.setting = setting
        self.mu_dim = mu_dim
        self.sigma_dim = sigma_dim
        self.dropout_rate = dropout_rate

        # --- Load ViT backbone ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.backbone = create_model(model_name, pretrained=True)
            embed_dim = self.backbone.embed_dim  # e.g., 768 for ViT-B/16

        # --- Register MC Dropout hooks (only on MLP layers for MCD) ---
        if self.setting in ['mcd']:
            self._register_mc_dropout_hooks(self.backbone, p=dropout_rate)

        # --- Mean Head (uses GGeM) ---
        self.mean_head = nn.Sequential(
            GGeM(num_channels=embed_dim, num_groups=num_groups),
            nn.Linear(embed_dim, mu_dim),
            L2Norm(dim=1),
        )

        # --- Sigma Head (for BTL or DUL) ---
        if self.setting in ['btl', 'dul']:
            self.sigma_head = nn.Sequential(
                GGeM(num_channels=embed_dim, num_groups=num_groups),
                nn.Linear(embed_dim, 500),
                nn.ReLU(),
                nn.Linear(500, sigma_dim, bias=True),
                nn.Softplus(),
            )
            # init sigma weights
            self.sigma_head[-2].weight.data.zero_()
            self.sigma_head[-2].bias.data.copy_(torch.log(torch.tensor(sigma_init)))
        else:
            self.sigma_head = None

    # --- MC Dropout Hook Registration ---
    def _register_mc_dropout_hooks(self, model, p):
        """Add dropout hooks to MLP linear layers only (not attention)."""
        def mc_dropout_hook(module, inputs, output):
            return F.dropout(output, p=p, training=True)

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'mlp' in name.lower():
                module.register_forward_hook(mc_dropout_hook)

    # --- Forward Pass ---
    def forward(self, x):
        # ViT feature extraction
        features = self.backbone.forward_features(x)  # [B, N, D]

        if features.ndim == 3:
            # remove CLS token
            features = features[:, 1:, :]  # [B, N-1, D]

        # Mean head (GGeM handles pooling internally)
        mu = self.mean_head(features)

        # Uncertainty heads
        if self.setting in ['btl', 'dul']:
            sigma = self.sigma_head(features)
            if self.setting == 'btl':
                return mu, sigma
            elif self.setting == 'dul':
                return reparametrize_trick(mu, sigma), [mu, sigma]
        elif self.setting in ['mcd', 'triplet']:
            return mu, torch.zeros((mu.shape[0], self.sigma_dim), device=mu.device)


# ---------------------- Test Run ----------------------

if __name__ == '__main__':
    model = Model(setting='mcd', model_name='vit_base_patch16_224.dino', num_groups=8)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)

    print("Output shapes:")
    if isinstance(out, tuple):
        for o in out:
            if isinstance(o, list):
                print([oo.shape for oo in o])
            else:
                print(o.shape)
    else:
        print(out.shape)
