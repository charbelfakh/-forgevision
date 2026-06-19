"""
Greedy coreset subsampling for PatchCore memory banks.

PatchCore (Roth et al., 2022) stores a subsampled set of normal patch embeddings.
Greedy k-center selection approximates the full training set in O(k·N) distance
computations — much cheaper than storing every patch on an 8 GB GPU.
"""

from __future__ import annotations

import torch


def greedy_coreset_indices(
    features: torch.Tensor,
    n_samples: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    Select n_samples indices via iterative furthest-point sampling.

    Args:
        features: (N, D) L2-normalised patch embeddings (CPU or GPU)
        n_samples: target coreset size (~1% of N by default in config)
        seed: RNG seed for the first centre

    Returns:
        (n_samples,) long tensor of selected row indices
    """
    n_total = features.shape[0]
    n_samples = min(n_samples, n_total)
    if n_samples == n_total:
        return torch.arange(n_total, device=features.device)

    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    first = torch.randint(0, n_total, (1,), generator=generator, device=features.device).item()

    selected: list[int] = [first]
    min_distances = torch.full((n_total,), float("inf"), device=features.device)

    for _ in range(n_samples - 1):
        centre = features[selected[-1] : selected[-1] + 1]
        dist = torch.cdist(features, centre, p=2).squeeze(1)
        min_distances = torch.minimum(min_distances, dist)
        next_idx = int(min_distances.argmax().item())
        selected.append(next_idx)

    return torch.tensor(selected, dtype=torch.long, device=features.device)
