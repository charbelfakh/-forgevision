"""
Batched nearest-neighbour search for PatchCore scoring (no FAISS).

Memory tradeoffs on 8 GB VRAM:
  - Keep the coreset memory bank on GPU when it fits (~few MB at 1% subsampling).
  - Process query patches in chunks (query_chunk) to limit peak VRAM during cdist.
  - If OOM persists, set use_gpu=False to run distance math on CPU (slower but safe).
"""

from __future__ import annotations

import torch


def nearest_neighbour_distances(
    queries: torch.Tensor,
    memory: torch.Tensor,
    *,
    query_chunk: int = 2048,
    memory_chunk: int = 4096,
    use_gpu: bool = True,
) -> torch.Tensor:
    """
    For each query vector, return L2 distance to its nearest neighbour in memory.

    Args:
        queries: (Nq, D)
        memory:  (Nm, D)
        query_chunk:  max queries per cdist call
        memory_chunk: max memory rows per cdist call
        use_gpu:      if False, compute on CPU regardless of tensor device

    Returns:
        (Nq,) minimum distances
    """
    if queries.numel() == 0:
        return torch.empty(0, device=queries.device)

    compute_device = queries.device if use_gpu else torch.device("cpu")
    queries = queries.to(compute_device)
    memory = memory.to(compute_device)

    nq = queries.shape[0]
    min_dists = torch.full((nq,), float("inf"), device=compute_device)

    for q_start in range(0, nq, query_chunk):
        q_end = min(q_start + query_chunk, nq)
        q_batch = queries[q_start:q_end]
        batch_min = torch.full((q_batch.shape[0],), float("inf"), device=compute_device)

        for m_start in range(0, memory.shape[0], memory_chunk):
            m_end = min(m_start + memory_chunk, memory.shape[0])
            m_batch = memory[m_start:m_end]
            dist = torch.cdist(q_batch, m_batch, p=2)
            batch_min = torch.minimum(batch_min, dist.min(dim=1).values)

        min_dists[q_start:q_end] = batch_min

    return min_dists.to(queries.device)
