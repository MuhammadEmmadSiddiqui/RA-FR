#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import cirtorch.functional as LF
from timm import create_model
import warnings


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps)


def reparametrize_trick(mu, sigma):
    sigma = sigma.sqrt()
    eps = torch.randn_like(sigma)
    return mu + eps * sigma


class Model(nn.Module):
    def __init__(self, mu_dim=768, sigma_dim=1, sigma_init=1e-3,
                 dropout_rate=0.1, setting='btl', model_name='vit_base_patch16_224.dino'):
        """
        ViT Backbone version of your model
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
            embed_dim = self.backbone.embed_dim  # typically 768 for base ViT

        # --- Register MC Dropout hooks (only on MLP layers for MCD) ---
        if self.setting in ['mcd']:
            self._register_mc_dropout_hooks(self.backbone, p=dropout_rate)

        # --- Mean Head ---
        self.mean_head = nn.Sequential(
            GeM(),
            nn.Flatten(),
            nn.Linear(embed_dim, mu_dim),
            L2Norm(dim=1),
        )

        # --- Sigma Head (only for BTL or DUL) ---
        if self.setting in ['btl', 'dul']:
            self.sigma_head = nn.Sequential(
                GeM(),
                nn.Flatten(),
                nn.Linear(embed_dim, 500),
                nn.ReLU(),
                nn.Linear(500, sigma_dim, bias=True),
                nn.Softplus(),
            )
            # initialize final linear to produce small positive sigma by default
            self.sigma_head[-2].weight.data.zero_()
            self.sigma_head[-2].bias.data.copy_(torch.log(torch.tensor(sigma_init)))
        else:
            self.sigma_head = None

    def _register_mc_dropout_hooks(self, model, p):
        """Add dropout hooks to MLP linear layers only (not attention)"""
        def mc_dropout_hook(module, inputs, output):
            return F.dropout(output, p=p, training=True)

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'mlp' in name.lower():
                module.register_forward_hook(mc_dropout_hook)

    def forward(self, x):
        # --- ViT feature extraction ---
        features = self.backbone.forward_features(x)  # expected [B, N, D] or [B, D]

        # If backbone returns token sequence (B, N, D) keep the non-CLS tokens and pool them
        if features.ndim == 3:
            # Remove CLS token (assumes CLS at index 0)
            features_tokens = features[:, 1:, :]  # [B, N-1, D]

            # Apply GeM pooling manually (over tokens) using the same p as mean_head's GeM
            p = torch.clamp(self.mean_head[0].p, min=1e-6)
            # gem pooling across token dim -> shape [B, D]
            features = (features_tokens.clamp(min=1e-6).pow(p)).mean(dim=1).pow(1.0 / p)  # [B, D]
        else:
            # fallback (in case forward_features returns already [B, D])
            # ensure features is [B, D]
            features = features

        # Defensive check: features must now be (B, D)
        if features.dim() != 2:
            raise ValueError(f"Expected pooled features of shape [B, D], got {features.shape}")

        # --- Heads ---
        # Use the pooled features directly for both heads (skip GeM/Flatten in the heads)
        mu = self.mean_head[2:](features)  # Linear(embed_dim->mu_dim) + L2Norm -> [B, mu_dim]

        if self.setting in ['btl', 'dul']:
            # --- FIXED: use pooled 'features' directly for sigma computation ---
            sigma = self.sigma_head[2:](features)  # Linear(embed_dim->500)->...->Softplus -> [B, sigma_dim]

            # make sure sigma is 2D (B, sigma_dim)
            if sigma.dim() == 1:
                sigma = sigma.unsqueeze(1)

            # final sanity check shape
            if sigma.shape[1] != self.sigma_dim:
                # try to reshape if possible, otherwise raise for easier debugging
                try:
                    sigma = sigma.view(sigma.shape[0], self.sigma_dim)
                except Exception:
                    raise ValueError(f"sigma has wrong shape {sigma.shape}, expected (_, {self.sigma_dim})")

            if self.setting == 'btl':
                return mu, sigma
            elif self.setting == 'dul':
                return reparametrize_trick(mu, sigma), [mu, sigma]
        elif self.setting in ['mcd', 'triplet']:
            return mu, torch.zeros((mu.shape[0], self.sigma_dim), device=mu.device)


#%%
if __name__ == '__main__':
    # quick smoke tests
    for setting in ['mcd', 'btl', 'dul']:
        print(f"\nTesting setting: {setting}")
        model = Model(setting=setting, model_name='vit_base_patch16_224.dino')
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        if isinstance(out, tuple):
            for o in out:
                if isinstance(o, list):
                    print([oo.shape for oo in o])
                else:
                    print(o.shape)
        else:
            print(out.shape)

    print('\nAll smoke tests completed.')
