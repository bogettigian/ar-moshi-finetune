from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def trust_torch_checkpoints() -> None:
    original = torch.load
    if getattr(original, "_trusts_pickled_checkpoints", False):
        return

    def trusted(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    trusted._trusts_pickled_checkpoints = True
    torch.load = trusted
    logger.debug("torch.load set to weights_only=False for pyannote checkpoints")
