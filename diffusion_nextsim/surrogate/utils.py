#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 01/02/2024
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2024}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Tuple, List

# External modules
import torch

# Internal modules

main_logger = logging.getLogger(__name__)


def split_wd_params(
        model: torch.nn.Module
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    # From minGPT https://github.com/karpathy/minGPT
    # Explanation: https://github.com/karpathy/minGPT/pull/24
    decay_params = set()
    no_decay_params = set()
    no_grad_params = set()
    for name, param in model.named_parameters():
        parent_module = model.get_submodule(".".join(name.split(".")[:-1]))
        decay = (
            name.endswith('weight')
            and not isinstance(parent_module, torch.nn.GroupNorm)
            and "norm" not in name
            and "embedding" not in name
            and "embedder" not in name
        )
        if decay and param.requires_grad:
            decay_params.add(name)
        elif param.requires_grad:
            no_decay_params.add(name)
        else:
            no_grad_params.add(name)

    # Check if all parameters are considered
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay_params & no_decay_params & no_grad_params
    union_params = decay_params | no_decay_params | no_grad_params
    missing_keys = param_dict.keys() - union_params
    if len(inter_params) != 0:
        raise AssertionError(
            "Parameters {0:s} made it into different sets!".format(
                str(inter_params)
            )
        )
    if len(missing_keys) != 0:
        raise AssertionError(
            "Parameters {0:s} were not separated into sets!".format(
                missing_keys
            )
        )

    # Convert into lists of parameters
    decay_params = [param_dict[pn] for pn in sorted(list(decay_params))]
    no_decay_params = [param_dict[pn] for pn in sorted(list(no_decay_params))]
    return decay_params, no_decay_params
