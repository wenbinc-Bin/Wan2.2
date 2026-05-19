# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging

import torch


def load_wan_model_low_cpu_mem(model_cls, checkpoint_dir, *, torch_dtype=None,
                               subfolder=None):
    kwargs = {'low_cpu_mem_usage': True}
    if torch_dtype is not None:
        kwargs['torch_dtype'] = torch_dtype
    if subfolder is not None:
        kwargs['subfolder'] = subfolder
    try:
        return model_cls.from_pretrained(checkpoint_dir, **kwargs)
    except TypeError as exc:
        if 'low_cpu_mem_usage' not in str(exc):
            raise
        logging.warning(
            'Model loader does not support low_cpu_mem_usage; falling back to '
            'the default loading path.')
        kwargs.pop('low_cpu_mem_usage', None)
        return model_cls.from_pretrained(checkpoint_dir, **kwargs)


def module_device(module):
    for tensor in module.parameters(recurse=True):
        return tensor.device
    for tensor in module.buffers(recurse=True):
        return tensor.device
    return torch.device('cpu')


def clear_device_cache(device=None):
    gc.collect()
    if device is None:
        return
    device = torch.device(device)
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == 'hpu' and hasattr(torch, 'hpu'):
        empty_cache = getattr(torch.hpu, 'empty_cache', None)
        if empty_cache is not None:
            empty_cache()


def synchronize_device(device):
    device = torch.device(device)
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == 'hpu' and hasattr(torch, 'hpu'):
        synchronize = getattr(torch.hpu, 'synchronize', None)
        if synchronize is not None:
            synchronize()


def move_module_to_device(module, device):
    module.to(device)
    synchronize_device(device)
    clear_device_cache(device)
    return module
