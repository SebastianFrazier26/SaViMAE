import torch
from timm.optim import create_optimizer_v2, optimizer_kwargs


def create_optimizer(args, model: torch.nn.Module):
    """
    Thin wrapper around timm.create_optimizer_v2 that matches the expected
    VideoMAE interface.

    VideoMAE typically passes an argparse.Namespace 'args' containing fields like:
      - opt (str): optimizer name, e.g. 'adamw'
      - lr (float)
      - weight_decay (float)
      - momentum, etc.

    We let timm.optim.optimizer_kwargs read from args and produce the correct
    keyword arguments for create_optimizer_v2().
    """
    # Let timm extract the optimizer kwargs from the args namespace
    opt_args = optimizer_kwargs(cfg=args)
    # Create the optimizer for the given model
    optimizer = create_optimizer_v2(model, **opt_args)
    return optimizer
