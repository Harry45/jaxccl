"""
Code: Different functions for computing the predictions from the trained GPs
Author: Dr. Arrykrishna Mootoovaloo
Collaborators: David, Carlos, Jaime
Date: June 2023
"""
from dataclasses import dataclass, field
from typing import List, Dict
import jax.numpy as jnp
from jax_cosmo.utils import load_pkl


@dataclass
class EMUdata:
    nz: int = 20
    nk: int = 30
    kmin: float = 1e-4
    kmax: float = 50.0
    zmin: float = 0.0
    zmax: float = 3.0
    path_quant: str = "jax_cosmo/quantities"

    quant_gf: List = field(init=False)
    quant_pk: List = field(init=False)
    zgrid: jnp.ndarray = field(init=False)
    kgrid: jnp.ndarray = field(init=False)

    priors: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "sigma8": {"distribution": "uniform", "loc": 0.6, "scale": 0.4},
        "Omega_cdm": {"distribution": "uniform", "loc": 0.07, "scale": 0.43},
        "Omega_b": {"distribution": "uniform", "loc": 0.028, "scale": 0.027},
        "h": {"distribution": "uniform", "loc": 0.64, "scale": 0.18},
        "n_s": {"distribution": "uniform", "loc": 0.87, "scale": 0.2},
    })

    def __post_init__(self):
        self.quant_gf = [load_pkl(self.path_quant, f"gf_{i}") for i in range(self.nz - 1)]
        self.quant_pk = [load_pkl(self.path_quant, f"pklin_{i}") for i in range(self.nk)]
        self.zgrid = jnp.linspace(self.zmin, self.zmax, self.nz)
        self.kgrid = jnp.geomspace(self.kmin, self.kmax, self.nk)

@dataclass
class EMUCMBdata:
    path_quant: str = "jax_cosmo/quantitiesCMB"
    ncomponents: int = 50
    ellmax: int = 2500

    quant_tt: List = field(init=False)
    quant_te: List = field(init=False)
    quant_ee: List = field(init=False)

    priors: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "sigma8": {"distribution": "uniform", "loc": 0.7, "scale": 0.2, "fiducial": 0.8},
        "Omega_cdm": {"distribution": "uniform", "loc": 0.20, "scale": 0.15, "fiducial": 0.2},
        "Omega_b": {"distribution": "uniform", "loc": 0.04, "scale": 0.02, "fiducial": 0.045},
        "h": {"distribution": "uniform", "loc": 0.62, "scale": 0.12, "fiducial": 0.68},
        "n_s": {"distribution": "uniform", "loc": 0.90, "scale": 0.2, "fiducial": 1.0},
    })

    def __post_init__(self):
        self.quant_tt = [load_pkl(self.path_quant, f"cmb_cls_tt_{i}") for i in range(self.ncomponents)]
        self.quant_te = [load_pkl(self.path_quant, f"cmb_cls_te_{i}") for i in range(self.ncomponents)]
        self.quant_ee = [load_pkl(self.path_quant, f"cmb_cls_ee_{i}") for i in range(self.ncomponents)]

        self.pipeline_tt = load_pkl('pipeline', 'cmb_cls_tt')
        self.pipeline_te = load_pkl('pipeline', 'cmb_cls_te')
        self.pipeline_ee = load_pkl('pipeline', 'cmb_cls_ee')
        self.ells = jnp.arange(2, self.ellmax + 1)


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

def prediction_cmb_cls(cosmology: jnp.ndarray, emudata: EMUCMBdata, cls_type: str = 'tt'):
    """Predicts the Cosmic Microwave Background (CMB) power spectrum using trained Gaussian Processes (GPs).

    Args:
        cosmology (np.ndarray): An array containing cosmological parameters.
        emudata (EMUCMBdata): An EMUCMBdata object containing trained GPs and preprocessing pipelines.
        cls_type (str): The type of CMB power spectrum to predict (e.g., 'tt', 'ee', 'te').

    Returns:
        np.ndarray: Predicted CMB power spectrum for the given cosmology.
    """
    if cls_type == 'tt':
        quant = emudata.quant_tt
        pipeline = emudata.pipeline_tt
    elif cls_type == 'te':
        quant = emudata.quant_te
        pipeline = emudata.pipeline_te
    elif cls_type == 'ee':
        quant = emudata.quant_ee
        pipeline = emudata.pipeline_ee
    else:
        raise ValueError("Invalid CMB power spectrum type. Choose from 'tt', 'ee', 'te'.")

    prediction = prediction_jax(cosmology, quant)
    prediction = pipeline.inverse_transform(prediction)
    return prediction
