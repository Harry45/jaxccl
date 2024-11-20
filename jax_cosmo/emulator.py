"""
Code: Different functions for computing the predictions from the trained GPs
Author: Dr. Arrykrishna Mootoovaloo
Collaborators: David, Carlos, Jaime
Date: June 2023
"""
import jax.numpy as jnp


def pairwise_distance_jax(arr1: jnp.ndarray, arr2: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the pairwise distance between two sets of matrices/vectors. If p is the
    dimensionality of the vector, both arr1 and arr2 must be of size (-1, p).

    Args:
        arr1 (jnp.ndarray): the first matrix of size (N1, p)
        arr2 (jnp.ndarray): the second matrix of size (N2, p)

    Returns:
        jnp.ndarray: a matrix containing the squared euclidean distance, of size (N1, N2)
    """
    sqr_a = jnp.broadcast_to(
        jnp.sum(jnp.power(arr1, 2), axis=1, keepdims=True),
        (arr1.shape[0], arr2.shape[0]),
    )
    sqr_b = jnp.broadcast_to(
        jnp.sum(jnp.power(arr2, 2), axis=1, keepdims=True),
        (arr2.shape[0], arr1.shape[0]),
    ).T
    dist = sqr_a - 2 * arr1 @ arr2.T + sqr_b
    return dist


def compute_jax(
    arr1: jnp.ndarray, arr2: jnp.ndarray, hyper: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculates the kernel matrix given two sets of points. The kernel employed is a radial basis kernel,
    and it has (1+p) hyperparameters (1 amplitude parameter and p lengthscale parameters).

    Args:
        arr1 (jnp.ndarray): the first set of points of size (N1, p)
        arr2 (jnp.ndarray): the second set of points of size (N2, p)
        hyper (jnp.ndarray): the kernel hyperparameters.

    Returns:
        jnp.ndarray: the kernel matrix of size (N1, N2)
    """
    _, ndim = arr1.shape
    arr2 = arr2.reshape(-1, ndim)
    hyper = hyper.reshape(1, ndim + 1)
    arr1 = arr1 / jnp.exp(hyper[:, 1:])
    arr2 = arr2 / jnp.exp(hyper[:, 1:])
    dist = pairwise_distance_jax(arr1, arr2)
    kernel = jnp.exp(hyper[:, 0]) * jnp.exp(-0.5 * dist)
    return kernel


def xtransform_jax(
    xtest: jnp.ndarray, cholfactor: jnp.ndarray, meanparams: jnp.ndarray
) -> jnp.ndarray:
    """
    Given the Cholesky factor and the mean (computed from the training set), this function
    scales the test point such that:

    x_prime = L^-1  (x_test - x_mean)

    We need this transformation because the GP is trained on the transformed training points.

    Args:
        xtest (jnp.ndarray): the test point
        cholfactor (jnp.ndarray): the Cholesky factor computed using the training set.
        meanparams (jnp.ndarray): the mean of the training set.

    Returns:
        jnp.ndarray: the transformed (rotated) parameter.
    """
    xtest = xtest.reshape(1, -1)
    xtest_trans = jnp.linalg.inv(cholfactor) @ (xtest - meanparams).T
    xtest_trans = xtest_trans.T
    return xtest_trans


def prediction_jax(xtest: jnp.ndarray, quantities: list) -> jnp.ndarray:
    """
    Predict the function using the trained GPs.

    Args:
        xtest (jnp.ndarray): the test point we want to use for prediction
        quantities (list): a list containing pkl files, where each file contains the following quantities:
        - hyperparams: hyperparameters for the kernel
        - cholfactor: the cholesky factor computed using the training set
        - meanparams: the mean parameter from the training set
        - alpha: alpha = Kinv y, see literature in Gaussian Process
        - ystd: the standard deviation of the outputs
        - ymean: the mean of the outputs (we need ystd and ymean because we centre the outputs on zero)
        - xtrain: the transformed training points

    Returns:
        jnp.ndarray: the predicted function
    """
    # the cosmologies are fixed and the cholesky factor is the same for all
    xtest_trans = xtransform_jax(
        xtest, quantities[0]["cholfactor"], quantities[0]["meanparams"]
    )
    ngps = len(quantities)
    predictions = []
    for i in range(ngps):
        kstar = compute_jax(
            quantities[i]["xtrain"], xtest_trans, quantities[i]["hyperparams"]
        )
        pred = kstar.T @ quantities[i]["alpha"]
        pred = pred * quantities[i]["ystd"] + quantities[i]["ymean"]
        predictions.append(pred)
    predictions = jnp.asarray(predictions).reshape(-1)
    return predictions


def prediction_pklin_jax(xtest: jnp.ndarray, quantities: list) -> jnp.ndarray:
    """
    Predicts the linear matter spectrum (at redshift 0) using the trained GPs.
    Because we trained on log P_lin, we have to transform back.

    Args:
        xtest (jnp.ndarray): the test point at which we want to predict the function.
        quantities (list): a list of the saved quantities (see prediction_jax function above)

    Returns:
        jnp.ndarray: the linear matter power spectrum
    """
    ypred = prediction_jax(xtest, quantities)
    return jnp.exp(ypred)


def prediction_gf_jax(xtest: jnp.ndarray, quantities: list) -> jnp.ndarray:
    """
    Predicts the growth factor

    Args:
        xtest (jnp.ndarray): the test point at which we want to compute the growth factor.
        quantities (list): a list of the stored quantities

    Returns:
        jnp.ndarray: the predicted growth factor
    """
    ypred = prediction_jax(xtest, quantities)

    # we concatenate one because the growth factor is 1 at redshift 0.
    ypred = jnp.concatenate([jnp.ones(1), ypred])
    return ypred
