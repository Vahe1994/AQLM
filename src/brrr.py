"""Utility functions for reduced rank regression"""

from typing import Optional

import torch


def reduced_rank_regression_from_data(
    X: torch.Tensor,
    Y: torch.Tensor,
    rank: int,
    *,
    reg_rate: float = 0,
    svd_niter: Optional[int] = 100,
    fit_intercept: bool = True,
    pinverse_device: Optional[torch.device] = None,
):
    """
    Multivariate linear regression with a low-rank weight matrix, based on
    https://web.math.ku.dk/~sjo/papers/ReducedRankRegression.pdf

    :note: this function computes RRR from raw data (x/y). If you have a pre-existing weight, it is much cheaper to
        reconstruct RRR directly from weight and the X.T@X matrix. See reduced_rank_regression_from_weight below.

    :param X: input, shape: [nsamples, in_features]
    :param Y: target, shape: [nsamples, num_targets]
    :param rank: the required rank of the weight matrix
    :param reg_rate: regularize weights by this value, in proportion to mean square input
    :param svd_niter: if specified, estimate SVD with this many steps of Algorithm 5.1 from Halko et al, 2009
        If None, compute SVD exactly and take the first k components
    :note: you can also compute partial SVD with arpack from scipy.sparse.linalg.svds; this is not implemented
    :param fit_intercept: if True (default), learn the bias term as in the regular least-squares
        if False, force bias term to be zeros -- but still return the resulting all-zero vector
    :param pinverse_device: optionally transfer the covariance matrix to this device and compute inverse there
        Computing pinverse of large matrices on CPU can take hours
    :returns: (U, V, b) such that Y ~ (X @ U @ V.T + b)


    :note: on using sample weights -- you can incorporate weights with pre/post-processing steps
      - when using per-dimension weights of shape [out_features], multiply only Y by dim_weight
        ... and then, divide both second matrix (VT) and intercept by dim_weight
      - when using per-sample weights of shape [nsamples], multiply both X and Y by
        ... sqrt(sample_weight)[:, None] / mean(sqrt(sample_weight)), and you should be fine
      - when using per-item weights of shape [nsamples, out_features], you will probably need SGD,
        ... consider starting from a non_weighted solution (or use 1d weights), then fine-tune with SGD
    """
    assert X.ndim == Y.ndim == 2 and X.shape[0] == Y.shape[0], "X, Y must be [batch_size, in/out_features]"
    assert rank <= min(X.shape[1], Y.shape[1]), "rank must be less than num features / outputs"
    (batch_size, in_features), (_, out_features) = X.shape, Y.shape

    CXX = X.T @ X  # [in_features, in_features], aka MSE hessian
    if reg_rate > 0:
        ix = torch.arange(len(CXX), device=X.device)
        CXX[ix, ix] += reg_rate * abs(torch.diag(CXX)).mean()
        del ix

    CXX_pinv = torch.pinverse(CXX.to(pinverse_device)).to(X.device)
    del CXX  # note: CXX can be computed on GPU by accumulating Xbatch.T@Xbatch products

    intercept = Y.mean(0) if fit_intercept else torch.zeros(out_features, dtype=X.dtype, device=X.device)
    CXY = X.T @ (Y - intercept)  # [in_features, out_features]
    del Y
    A = torch.linalg.multi_dot((CXY.T, CXX_pinv, CXY))  # [out_features, out_features]
    if svd_niter is not None:
        _, _, V = torch.svd_lowrank(A, q=rank, niter=svd_niter)
    else:
        # Note: since A is symmetric, we replace SVD with eigendecomposition, which is faster;
        # The eigenvectors will equal either the columns of V (from SVD), or columns of (-1 * V), which is okay for RRR
        _unused_eigvals, eigvecs = torch.linalg.eigh(A)  # this returns eigenvecs/vals in *ascending* order
        V = eigvecs[:, torch.arange(in_features - 1, in_features - 1 - rank, -1)].contiguous()  # take *rank* largest
        del _unused_eigvals, eigvecs
    # VT: [out_features, rank]
    W = torch.linalg.multi_dot((CXX_pinv, CXY, V))
    return W, V.T.contiguous(), intercept


def reduced_rank_regression_from_weight(
    XTX: torch.Tensor,
    W: torch.Tensor,
    rank: int,
    *,
    svd_niter: Optional[int] = 100,
):
    """
    Multivariate linear regression with a low-rank weight matrix, based on
    https://web.math.ku.dk/~sjo/papers/ReducedRankRegression.pdf

    This is a special case that relies on pre-existing full-rank weight matrix to compute RRR faster and more accurately

    :param XTX: covariance matrix aka mse hessian, shape: [in_features, in_features]
    :param W: weight, shape: [out_features, in_features]
    :param rank: the required rank of the weight matrix
    :param svd_niter: if specified, estimate SVD with this many steps of Algorithm 5.1 from Halko et al, 2009
        If None, compute SVD exactly and take the first k components
    :note: you can also compute partial SVD with arpack from scipy.sparse.linalg.svds; this is not implemented
    :returns: (U, V, b) such that Y ~ X @ U @ V.T
       In other words, U @ V.T ~ W.T    or equivalently W ~ V @ U.T

    :note: on using sample weights -- you can incorporate weights with pre/post-processing steps
      - when using per-dimension weights of shape [out_features], multiply only Y by dim_weight
        ... and then, divide both second matrix (VT) and intercept by dim_weight
      - when using per-sample weights of shape [nsamples], multiply both X and Y by
        ... sqrt(sample_weight)[:, None] / mean(sqrt(sample_weight)), and you should be fine
      - when using per-item weights of shape [nsamples, out_features], you will probably need SGD,
        ... consider starting from a non_weighted solution (or use 1d weights), then fine-tune with SGD
    """

    assert XTX.ndim == 2 and XTX.shape[0] == XTX.shape[1], "XTX must be [in_features, in_features]"
    assert W.ndim == 2 and W.shape[1] == XTX.shape[1]
    assert rank <= min(W.shape[0], W.shape[1]), "rank must be less than num features / outputs"
    (out_features, in_features) = W.shape

    # A = YTX inv(XTX) XTY = YTX [inv(XTX) XTX] inv(XTX) XTY = W XTX W.T
    A = torch.linalg.multi_dot((W, XTX, W.T))  # [out_features, out_features]

    if svd_niter is not None:
        _, _, V = torch.svd_lowrank(A, q=rank, niter=svd_niter)
    else:
        _, _, VT = torch.linalg.svd(A)
        V = VT[:rank, :].T.contiguous()
        del VT
    # VT: [out_features, rank]
    U = torch.linalg.multi_dot((W.T, V))
    return U, V
