# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist


def init_distributed_group():
    """r initialize sequence parallel group.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = get_world_size()
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


def all_gather(tensor):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return tensor_list


def gather_forward(input, dim):
    # skip if world_size == 1
    world_size = dist.get_world_size()
    if world_size == 1:
        return input

    # gather sequence
    output = all_gather(input)
    return torch.cat(output, dim=dim).contiguous()


def all_gather_forward(tensor):
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    bs, kv_seq, num_head, head_dim = tensor.shape
    tensor = tensor.reshape(bs, kv_seq, -1)
    full_tensor = torch.empty(bs, kv_seq * world_size, num_head * head_dim, dtype=tensor.dtype, device=tensor.device)
    torch.distributed.all_gather_into_tensor(
        full_tensor,
        tensor,
        group=None,
        async_op=False,
    )
    tensor = full_tensor.reshape(bs, kv_seq * world_size, num_head, head_dim)
    return tensor
