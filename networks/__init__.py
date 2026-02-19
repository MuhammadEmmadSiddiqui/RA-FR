"""
Network Architecture Factory

This module provides a unified interface to select different model architectures
for the RAFR (Risk Aware Facial Retrieval) framework.

Available architectures:
- R50-GeM: ResNet50 with Generalized Mean Pooling (baseline from RCIR)
- ViTB-DN-GeM: Vision Transformer Base with DeiT normalization and GeM pooling
- ViTB-DN-GGeM: Vision Transformer Base with DeiT normalization and Group GeM pooling (proposed)

Usage:
    from networks import get_model
    
    model = get_model(
        arch='ViTB-DN-GGeM',
        mu_dim=2048,
        sigma_dim=1,
        setting='btl'
    )
"""

def get_model(arch='R50-GeM', **kwargs):
    """
    Factory function to create model instances based on architecture name.
    
    Args:
        arch: Architecture name. Options:
            - 'R50-GeM': ResNet50 with GeM pooling (baseline)
            - 'ViTB-DN-GeM': ViT-Base with DN and GeM pooling
            - 'ViTB-DN-GGeM': ViT-Base with DN and Group GeM pooling (proposed)
        **kwargs: Additional arguments passed to the model constructor
            - mu_dim: Dimension of mean embedding (default: 2048)
            - sigma_dim: Dimension of uncertainty (default: 1)
            - sigma_init: Initial sigma value (default: 1e-3)
            - dropout_rate: Dropout rate (default: 0.1)
            - setting: Training method 'btl' or 'mcd' (default: 'btl')
    
    Returns:
        Model instance
    
    Example:
        >>> model = get_model('ViTB-DN-GGeM', mu_dim=768, setting='btl')
        >>> model = get_model('R50-GeM', mu_dim=2048, setting='mcd')
    """
    arch_map = {
        'R50-GeM': 'networks.R50-GeM',
        'ViTB-DN-GeM': 'networks.ViTB-DN-GeM',
        'ViTB-DN-GGeM': 'networks.ViTB-DN-GGeM',
    }
    
    # Also support network.py for backward compatibility
    if arch == 'network' or arch == 'default':
        from networks.network import Model
        return Model(**kwargs)
    
    if arch not in arch_map:
        available = ', '.join(arch_map.keys())
        raise ValueError(
            f"Unknown architecture: {arch}\n"
            f"Available architectures: {available}"
        )
    
    # Dynamically import the model
    module_path = arch_map[arch]
    module_name = module_path.split('.')[-1]
    
    # Import using importlib for cleaner dynamic loading
    import importlib
    module = importlib.import_module(module_path)
    Model = module.Model
    
    return Model(**kwargs)


# Convenience exports
__all__ = ['get_model']
