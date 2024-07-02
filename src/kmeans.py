import itertools
from typing import List, Optional, Tuple

import torch

from src.utils import maybe_script


@maybe_script
def _kmeans_greedy_init(data: torch.Tensor, k: int) -> torch.Tensor:
    """Get initial clusters by iteratively choosing a vector that is the farthest from already selected clusters"""
    clusters = torch.zeros(k, data.shape[1], device=data.device)
    running_min_distances = torch.full((data.shape[0],), torch.inf, device=data.device, dtype=data.dtype)
    data_norm_squared = data.norm(p=2, dim=1).square()

    for i in range(k):
        clusters[i] = data[running_min_distances.argmax()]
        distances_to_cluster_i = data_norm_squared - 2 * data @ clusters[i] + clusters[i].norm().square()
        running_min_distances = torch.minimum(running_min_distances, distances_to_cluster_i, out=running_min_distances)
    return clusters


@maybe_script
def fit_kmeans(
    data: torch.Tensor,
    k: int,
    max_iter: int = 1000,
    check_every: int = 10,
    rtol: float = 1e-06,
    atol: float = 1e-08,
    greedy_init: bool = False,
    block_size_vals: int = 2**30,
    devices: Optional[List[torch.device]] = None,
):
    """
    :param data: [nsamples, dim]
    :param k: number of centroids
    :param max_iter: run at most this many iterations
    :param check_every: check for convergence (allclose(new_centroids, old_centroids)) once in this many steps
    :param rtol: early stopping relative tolerance for centroids
    :param atol: early stopping absolute tolerance for centroids
    :param greedy_init: if True, init by greedily selecting the point that is farthest from any cluster
        if False (default), initialize with random points using pytorch global RNG
    :param block_size_vals: how many dot products to compute at a time
    :param devices: if specified, run kmeans in data-parallel mode across these devices
    :return: (clusters float[k, dim], data_indices int[nsamples], reconstructed_data: float[nsamples, dim])
    """
    if devices is None:
        devices = [data.device]

    if greedy_init:
        clusters = _kmeans_greedy_init(data, k)
    else:
        clusters = data[torch.randperm(data.shape[0])[:k], :]  # [k, dim]

    block_size = block_size_vals // k
    shard_size = (len(data) - 1) // len(devices) + 1
    data = [
        data[gi * shard_size : (gi + 1) * shard_size].to(devices[gi], non_blocking=True) for gi in range(len(devices))
    ]
    nearest_indices = [torch.empty(len(data[gi]), dtype=torch.int64, device=devices[gi]) for gi in range(len(devices))]
    clusters = [clusters.to(device, non_blocking=True) for device in devices]

    for i in range(max_iter):
        for block_start in range(0, shard_size, block_size):
            for gi in range(len(devices)):
                nearest_indices[gi][block_start : block_start + block_size] = torch.addmm(
                    torch.bmm(clusters[gi][:, None, :], clusters[gi][:, :, None]).flatten(),
                    data[gi][block_start : block_start + block_size],
                    clusters[gi].T,
                    beta=-0.5,
                ).argmax(1)
            # note: the above formula equals to - 0.5 || data[:, None, :] - clusters[None, :, :] || ^ 2 + const

        if len(devices) == 1:
            new_clusters = [
                clusters[0]
                .clone()
                .index_reduce_(dim=0, index=nearest_indices[0], source=data[0], reduce="mean", include_self=False)
            ]
        else:
            cluster_sums = [
                torch.zeros_like(clusters[gi])
                .index_add(dim=0, index=nearest_indices[gi], source=data[gi])
                .to(devices[0], non_blocking=True)
                for gi in range(len(devices))
            ]
            cluster_counts = [
                torch.bincount(nearest_indices[gi], minlength=k).to(devices[0], non_blocking=True)
                for gi in range(len(devices))
            ]
            for gi in range(1, len(devices)):
                cluster_sums[0] += cluster_sums[gi]
                cluster_counts[0] += cluster_counts[gi]

            new_clusters = [cluster_sums[0] / cluster_counts[0].unsqueeze(1).clamp_min(1)]
            new_clusters[0] += (cluster_counts[0].unsqueeze(1) == 0) * clusters[0]
            for gi in range(1, len(devices)):
                new_clusters.append(new_clusters[0].to(devices[gi], non_blocking=True))

        if i % check_every == 0:
            if torch.allclose(new_clusters[0], clusters[0], rtol=rtol, atol=atol):
                break
        clusters = new_clusters
    for block_start in range(0, shard_size, block_size):
        for gi in range(len(devices)):
            nearest_indices[gi][block_start : block_start + block_size] = torch.addmm(
                torch.bmm(clusters[gi][:, None, :], clusters[gi][:, :, None]).flatten(),
                data[gi][block_start : block_start + block_size],
                clusters[gi].T,
                beta=-0.5,
            ).argmax(1)

    clusters = clusters[0]
    nearest_indices = torch.cat([nearest_indices[gi].to(devices[0]) for gi in range(len(devices))], dim=0)
    reconstructed_data = clusters[nearest_indices]
    return clusters, nearest_indices, reconstructed_data


def fit_faiss_kmeans(
    data: torch.Tensor,
    k: int,
    *,
    max_iter: int = 1000,
    gpu: bool = True,
    max_points_per_centroid: Optional[int] = None,
    verbose: bool = True,
):
    """
    :param data: [nsamples, dim]
    :param k: number of centroids
    :param max_iter: run at most this many iterations
    :param gpu: if True, run kmeans on (all available) GPUs; if False, run on CPU
    :param max_points_per_centroid: if specified, train kmeans on a random subset of (this_many * k) points
    :param verbose: if True, faiss.kmeans will print status to stdout

    :return: (clusters float[k, dim], data_indices int[nsamples], reconstructed_data: float[nsamples, dim])
    """
    try:
        import faiss
    except ModuleNotFoundError:
        raise RuntimeError("Faiss is not installed. Please install it before running this function.")

    d = data.shape[1]
    if max_points_per_centroid is not None:
        kmeans = faiss.Kmeans(
            d, k, niter=max_iter, verbose=verbose, gpu=gpu, max_points_per_centroid=max_points_per_centroid
        )
    else:
        kmeans = faiss.Kmeans(d, k, niter=max_iter, verbose=verbose, gpu=gpu)
    kmeans.train(data.cpu())
    clusters = kmeans.centroids
    nearest_indices = kmeans.index.search(data.cpu(), 1)[1][:, 0]
    clusters, nearest_indices = torch.from_numpy(clusters).to(data.device), torch.from_numpy(nearest_indices).to(
        data.device
    )
    reconstructed_data = clusters[nearest_indices]

    return clusters, nearest_indices, reconstructed_data


@maybe_script
def find_nearest_cluster(data, clusters, block_size_vals: int = 2**30, devices: Optional[List[torch.device]] = None):
    """Find nearest clusters for each batch of data and return their indices"""
    if devices is None:
        devices = [data.device]
    block_size = block_size_vals // len(clusters)
    shard_size = (len(data) - 1) // len(devices) + 1
    data = [
        data[gi * shard_size : (gi + 1) * shard_size].to(devices[gi], non_blocking=True) for gi in range(len(devices))
    ]
    nearest_indices = [torch.empty(len(data[gi]), dtype=torch.int64, device=devices[gi]) for gi in range(len(devices))]
    clusters = [clusters.to(device, non_blocking=True) for device in devices]

    for block_start in range(0, shard_size, block_size):
        for gi in range(len(devices)):
            nearest_indices[gi][block_start : block_start + block_size] = torch.addmm(
                torch.bmm(clusters[gi][:, None, :], clusters[gi][:, :, None]).flatten(),
                data[gi][block_start : block_start + block_size],
                clusters[gi].T,
                beta=-0.5,
            ).argmax(1)
    clusters = clusters[0]
    nearest_indices = torch.cat([nearest_indices[gi].to(devices[0]) for gi in range(len(devices))], dim=0)
    reconstructed_data = clusters[nearest_indices]
    return nearest_indices, reconstructed_data


def fit_kmeans_1d(
    groupwise_data: torch.Tensor,
    k: int,
    max_iter: int = -1,
    offset_rate: float = 0,
    verbose: bool = False,
    initial_clusters: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    optimized batch k-means for 1d datapoint using sort
    :param groupwise_data: stuff to be compressed, shape: [num_groups, group_size]
    :param k: the number of centroids to find
    :param max_iter: run for at most this many kmeans iterations (-1 = run until convergence)
    :param offset_rate: if greater than 0, skip this percentage of smallest/largest elements for initialization
    :param verbose: print mse and early stopping info
    :param kwargs: optionally provide rtol=... and atol=... for early stopping;
    :note: if rtol/atol is speficied, these tolerances are measured between cluster centroids from subsequent steps
    :returns: (clusters, indices, restored_data)
        - clusters are centroids of shape
        - indices are integers [0, k) in the same shape as data; they denote the index of the nearest centroid
        - restored_data is a floating point tensor in the same shape as data; they are dequantized(quantized(data))
    :note: to reconstruct clusters manually, call clusters.gather(-1, indices)
    :TODO[aqlm]: torch.jit.script / torch.compile
    """
    assert groupwise_data.ndim == 2
    assert 0 <= offset_rate < 0.5

    # step 2: pre-sort data and initialize kmeans with uniform percentiles
    sorted_data, groupwise_sort_indices = groupwise_data.sort(dim=1)
    groupwise_ranks_1based = groupwise_sort_indices.argsort(-1).add_(1)
    del groupwise_sort_indices

    # ^-- [num_groups, group_size]; sorted by group_size
    sorted_cumsum = torch.cat([torch.zeros_like(sorted_data[:, :1]), sorted_data.cumsum(dim=1)], dim=1)
    # ^-- [num_groups, group_size + 1]; sorted by group_size + 1
    if initial_clusters is not None:
        clusters = initial_clusters
    else:
        offset = int((sorted_data.shape[1] - 1) * offset_rate)
        init_indices = torch.linspace(offset, sorted_data.shape[1] - 1 - offset, k, dtype=torch.int64)
        clusters = sorted_data[:, init_indices]  # shape: [num_groups, k]

    # step 3: run kmeans
    def _groupwise_find_border_indices(clusters, sorted_data):
        borders = (clusters[:, 1:] + clusters[:, :-1]) / 2
        column = clusters[:, :1]
        borders = torch.cat(
            [torch.full_like(column, float("-inf")), borders, torch.full_like(column, float("inf"))], dim=1
        )
        border_indices = torch.searchsorted(sorted_data, borders, side="left")
        return border_indices

    for i in itertools.count():
        border_indices = _groupwise_find_border_indices(clusters, sorted_data)
        sum_by_cluster = torch.diff(sorted_cumsum.gather(1, border_indices), dim=1)
        count_by_cluster = torch.diff(border_indices, dim=1)
        new_cluster_centers = torch.where(
            count_by_cluster > 0,
            sum_by_cluster / count_by_cluster,
            sorted_data.gather(1, border_indices[:, :-1].clamp_max(sorted_data.shape[1] - 1)),
        )
        if torch.allclose(new_cluster_centers, clusters, **kwargs):
            if verbose:
                print(f"Early stopping after {i} iterations")
            break
        clusters = new_cluster_centers
        if max_iter > 0 and i >= max_iter:
            break

    # step 4: determine the final clustering
    border_indices = _groupwise_find_border_indices(clusters, sorted_data)
    groupwise_cluster_indices = torch.searchsorted(border_indices[:, 1:], groupwise_ranks_1based, side="left")
    groupwise_restored_data = clusters.gather(1, groupwise_cluster_indices)
    # [num_groups, k]

    if verbose:
        sorted_cumsum_squares = torch.cat(
            [torch.zeros_like(sorted_data[:, :1]), sorted_data.square().cumsum(dim=1)], dim=1
        )
        sum_by_cluster = torch.diff(sorted_cumsum.gather(1, border_indices), dim=1)
        sum_squares_by_cluster = torch.diff(sorted_cumsum_squares.gather(1, border_indices), dim=1)
        count_by_cluster = torch.diff(border_indices, dim=1).clamp_min(1)
        mse_l2 = (groupwise_restored_data - groupwise_data).square().mean()
        mse_approx = sum_squares_by_cluster - 2 * sum_by_cluster * clusters + count_by_cluster * clusters.square()
        mse_approx = mse_approx.sum(0) / count_by_cluster.sum(0)
        print(f"mse: {mse_l2.mean().item()} , dot-based estimate: {mse_approx.mean().item()}")

    return clusters, groupwise_cluster_indices, groupwise_restored_data
