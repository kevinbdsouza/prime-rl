from typing import Dict, List

import torch

import zeroband.training.envs as envs


class WorldInfo:
    """
    This class retrieves topology information for distributed training and inference settings by parsing environment variables, typically set by torchrun.
    """

    rank: int
    world_size: int
    local_rank: int
    local_world_size: int

    def __init__(
        self,
        rank: int | None = None,
        world_size: int | None = None,
        local_rank: int | None = None,
        local_world_size: int | None = None,
        node_group_sizes: List[int] | None = None,
    ):
        """
        Initialize the WorldInfo object either manually or by parsing environment variables.
        """
        self.rank = rank if rank is not None else envs.RANK
        self.world_size = world_size if world_size is not None else envs.WORLD_SIZE

        # Support uneven node groups via NODE_GROUP_SIZES env var or argument
        self.node_group_sizes = node_group_sizes or envs.NODE_GROUP_SIZES

        if self.node_group_sizes:
            assert sum(self.node_group_sizes) == self.world_size, "NODE_GROUP_SIZES must sum to WORLD_SIZE"
            cumulative = 0
            for idx, size in enumerate(self.node_group_sizes):
                if self.rank < cumulative + size:
                    self.local_rank = self.rank - cumulative
                    self.local_world_size = size
                    self.node_idx = idx
                    break
                cumulative += size
            else:
                raise AssertionError("Rank out of range of NODE_GROUP_SIZES")
            self.num_nodes = len(self.node_group_sizes)
        else:
            self.local_rank = local_rank if local_rank is not None else envs.LOCAL_RANK
            self.local_world_size = local_world_size if local_world_size is not None else envs.LOCAL_WORLD_SIZE
            self.num_nodes = self.world_size // self.local_world_size
            self.node_idx = self.rank // self.local_world_size

        self.gpu_ids = envs.CUDA_VISIBLE_DEVICES or list(range(torch.cuda.device_count()))
        self._check_world_info()
        self.num_gpus = len(self.gpu_ids)

    def _check_world_info(self):
        assert 0 <= self.local_rank < self.local_world_size
        assert 0 <= self.rank < self.world_size
        assert self.local_world_size <= self.world_size
        if self.node_group_sizes is None:
            assert self.world_size % self.local_world_size == 0
        else:
            assert sum(self.node_group_sizes) == self.world_size
            assert self.node_group_sizes[self.node_idx] == self.local_world_size

    def __repr__(self):
        return (
            "WorldInfo("
            f"world_size={self.world_size}, "
            f"rank={self.rank}, "
            f"local_rank={self.local_rank}, "
            f"local_world_size={self.local_world_size}, "
            f"num_nodes={self.num_nodes}, "
            f"num_gpus={self.num_gpus}, "
            f"gpu_ids={self.gpu_ids})"
        )

    def json(self) -> Dict[str, int]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "local_world_size": self.local_world_size,
            "num_nodes": self.num_nodes,
        }


# Singleton instance of WorldInfo
_WORLD_INFO: WorldInfo | None = None


def get_world_info(
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
    local_world_size: int | None = None,
    node_group_sizes: List[int] | None = None,
) -> WorldInfo:
    """Returns the WorldInfo. If not initialized, it will initialize."""
    global _WORLD_INFO
    if _WORLD_INFO is None:
        _WORLD_INFO = WorldInfo(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            node_group_sizes=node_group_sizes,
        )
    return _WORLD_INFO


def reset_world_info() -> None:
    global _WORLD_INFO
    _WORLD_INFO = None


if __name__ == "__main__":
    # Used in tests/units/test_world_info.py to test init with torchrun
    import torch.distributed as dist

    print(get_world_info())
    if dist.is_initialized():
        dist.destroy_process_group()
