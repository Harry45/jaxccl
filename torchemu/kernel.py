"""
Author: Dr. Arrykrishna Mootoovaloo
Date: 30th June 2022
Email: arrykrish@gmail.com, a.mootoovaloo17@imperial.ac.uk, arrykrishna.mootoovaloo@physics
Description: Functions related to the Gaussian Process
"""

# Author: Dr. Arrykrishna Mootoovaloo
# Date: 17th January 2022
# Email: arrykrish@gmail.com, a.mootoovaloo17@imperial.ac.uk, arrykrishna.mootoovaloo@physics
# Description: all calculations related to the kernel matrix

import torch


def pairwise_distance(arr1: torch.tensor, arr2: torch.tensor) -> torch.tensor:
    """Compute the pairwise distance between two sets of points.

    Args:
        x1 (torch.tensor): [N x d] tensor of points.
        x2 (torch.tensor): [M x d] tensor of points.

    Returns:
        torch.tensor: a tensor of size [N x M] containing the pairwise distance.
    """

    # compute pairwise distances
    # d = torch.cdist(x1, x2, p=2)
    # d = torch.pow(d, 2)
    # cdist does not support gradient (for hessian computation) so we use the following
    # https://nenadmarkus.com/p/all-pairs-euclidean/

    sqr_a = torch.sum(torch.pow(arr1, 2), 1, keepdim=True).expand(
        arr1.shape[0], arr2.shape[0]
    )
    sqr_b = (
        torch.sum(torch.pow(arr2, 2), 1, keepdim=True)
        .expand(arr2.shape[0], arr1.shape[0])
        .t()
    )
    dist = sqr_a - 2 * torch.mm(arr1, arr2.t()) + sqr_b

    return dist


def compute(
    arr1: torch.tensor, arr2: torch.tensor, hyper: torch.tensor
) -> torch.tensor:
    """Compute the kernel matrix between two sets of points.

    Args:
        x1 (torch.tensor): [N x d] tensor of points.
        x2 (torch.tensor): [M x d] tensor of points.
        hyper (torch.tensor): [d+1] tensor of hyperparameters.

    Returns:
        torch.tensor: a tensor of size [N x M] containing the kernel matrix.
    """

    _, ndim = arr1.shape

    # reshape all tensors in the right dimensions
    arr2 = arr2.view(-1, ndim)

    # for the hyperparameters, we have an amplitude and ndim lengthscales
    hyper = hyper.view(1, ndim + 1)

    # the inputs are scaled by the characteristic lengthscale
    arr1 = arr1 / torch.exp(hyper[:, 1:])
    arr2 = arr2 / torch.exp(hyper[:, 1:])

    # compute the pairwise distance
    dist = pairwise_distance(arr1, arr2)

    # compute the kernel
    kernel = torch.exp(hyper[:, 0]) * torch.exp(-0.5 * dist)

    return kernel


def solve(kernel: torch.tensor, vector: torch.tensor) -> torch.tensor:
    """Solve the linear system Kx = b.

    Args:
        kernel (torch.tensor): [N x N] kernel matrix.
        vector (torch.tensor): [N x 1] vector.

    Returns:
        torch.tensor: [N x 1] vector.
    """

    solution = torch.linalg.solve(kernel, vector)

    return solution


def logdeterminant(kernel: torch.tensor) -> torch.tensor:
    """Compute the log determinant of the kernel matrix.

    Args:
        kernel (torch.tensor): [N x N] kernel matrix.

    Returns:
        torch.tensor: [1 x 1] vector.
    """

    logdet = torch.logdet(kernel)

    return logdet
