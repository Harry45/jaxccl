import os
import jax
import torch
import numpy as np
import pandas as pd
import scipy.stats as ss
from typing import Tuple, List, Dict, Union
from ml_collections.config_dict import ConfigDict
import jax.numpy as jnp
from jax import jit

from cmbrun.cmbcls import calculate_cmb_cls_class
from torchemu.gaussianprocess import GaussianProcess
from jax_cosmo.utils import load_pkl, save_pkl

def generate_cosmo_priors(cfg: ConfigDict) -> Dict:
    """Generate cosmological priors based on the configuration settings.

    Args:
        cfg (ConfigDict): A configuration dictionary containing cosmological parameter names and their prior distributions.

    Returns:
        Dict: A dictionary where keys are cosmological parameter names and values are their corresponding prior distributions.
    """
    dictionary = dict()
    for name in cfg.cosmo.names:
        param = cfg.priors[name]
        specs = (param.loc, param.scale)
        dictionary[name] = getattr(ss, param.distribution)(*specs)
    return dictionary

def generate_inputs(lhs: pd.DataFrame, priors: Dict, save: bool = False, fname: str = 'cosmo_cmb') -> pd.DataFrame:
    """Generate the input training points (the cosmologies).

    This function scales the Latin hypercube samples according to the prior range of the cosmological parameters.

    Args:
        lhs (pd.DataFrame): A DataFrame containing Latin hypercube samples.
        priors (Dict): A dictionary where keys are cosmological parameter names and values are their corresponding prior distributions.
        save (bool, optional): If True, the generated cosmologies will be saved to a CSV file. Defaults to False.
        fname (str, optional): The filename for saving the generated cosmologies. Defaults to 'cosmo_cmb'.

    Returns:
        pd.DataFrame: A DataFrame containing the scaled cosmological parameters.

    """
    cosmologies = {}
    for i, p in enumerate(priors):
        cosmologies[p] = priors[p].ppf(lhs.iloc[:, i].values)
    cosmologies = pd.DataFrame(cosmologies)
    if save:
        os.makedirs('data', exist_ok=True)
        cosmologies.to_csv(f'data/{fname}.csv')
    return cosmologies

def generate_cmb_cls_outputs(cosmologies: pd.DataFrame,
                     cfg: ConfigDict,
                     save: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ncosmo = cosmologies.shape[0]
    record_tt = []
    record_ee = []
    record_te = []

    for i in range(ncosmo):
        cosmology = dict(cosmologies.iloc[i])
        ells, cls_tt, cls_ee, cls_te = calculate_cmb_cls_class(cosmology, cfg)
        record_tt.append(cls_tt)
        record_ee.append(cls_ee)
        record_te.append(cls_te)
    record_tt = pd.DataFrame(record_tt, columns=ells)
    record_ee = pd.DataFrame(record_ee, columns=ells)
    record_te = pd.DataFrame(record_te, columns=ells)
    if save:
        record_tt.to_csv('data/cmb_cls_tt.csv')
        record_ee.to_csv('data/cmb_cls_ee.csv')
        record_te.to_csv('data/cmb_cls_te.csv')
    return record_tt, record_te, record_ee

def train_gps(config: ConfigDict,
              cosmologies: torch.Tensor,
              outputs: torch.Tensor,
              prewhiten: bool,
              ylog: bool,
              fname: str='cmb_cls_tt_0') -> List:
    """Train Gaussian Processes (GPs) for the given cosmologies and outputs.

    This function optimizes the kernel parameters for each output dimension and saves the trained GP models.

    Args:
        config (ConfigDict): A configuration dictionary containing emulator settings.
        cosmologies (torch.Tensor): A tensor containing the cosmological parameters.
        outputs (torch.Tensor): A tensor containing the output values for each cosmology.
        prewhiten (bool): If True, prewhiten the data before training.
        ylog (bool): If True, apply a logarithmic transformation to the output values.
        fname (str): The base filename for saving the trained GP models. Defaults to 'cmb_cls_tt_0'.

    Returns:
        List[GaussianProcess]: A list of trained GaussianProcess models.

    """
    nout = outputs.shape[1]
    record = []
    for i in range(nout):

        # optimise for the kernel parameters
        gpmodule = GaussianProcess(config,
                                   cosmologies,
                                   outputs[:, i],
                                   prewhiten=prewhiten,
                                   ylog=ylog)
        parameters = torch.randn(6)
        opt_params = gpmodule.optimisation(parameters,
                                           niter=config.emu.niter,
                                           lrate=config.emu.lr,
                                           nrestart=config.emu.nrestart)

        # save the gps and quantities
        save_pkl(gpmodule, 'gps', fname + f'_{i}')
        record.append(gpmodule)
    return record

class JAXPreprocessingPipeline:
    def __init__(self, n_components=50, epsilon=1e-6, apply_log=False):
        """
        Initializes the preprocessing pipeline.
        Args:
            n_components (int): Number of principal components for PCA.
            epsilon (float): Small regularization constant for prewhitening.
            apply_log (bool): Whether to apply log-transformation to the data.
        """
        self.n_components = n_components
        self.epsilon = epsilon  # Regularization for prewhitening
        self.apply_log = apply_log  # Option to apply log-transformation

    def fit(self, X):
        """
        Fit the pipeline: log-transform (optional), scale the data, apply PCA, and prewhiten.
        Args:
            X (jnp.ndarray): Input data of shape (N, N_c).
        Returns:
            self
        """
        # Step 1: Apply log-transformation (if enabled)
        if self.apply_log:
            self.shift_ = jnp.abs(jnp.min(X)) + 1 if jnp.min(X) <= 0 else 0  # Shift for non-negative log
            X = jnp.log(X + self.shift_)

        # Step 2: Compute column-wise means and standard deviations
        self.means_ = jnp.mean(X, axis=0)
        self.stds_ = jnp.std(X, axis=0)
        X_scaled = (X - self.means_) / self.stds_

        # Step 3: Apply PCA using SVD
        U, S, Vt = jnp.linalg.svd(X_scaled, full_matrices=False)
        self.components_ = Vt[:self.n_components]  # Top principal components
        X_reduced = jnp.dot(X_scaled, self.components_.T)

        # Step 4: Compute covariance of reduced data and prewhiten
        cov_matrix = jnp.cov(X_reduced, rowvar=False)
        cov_matrix += jnp.eye(cov_matrix.shape[0]) * self.epsilon  # Regularization
        self.L_ = jnp.linalg.cholesky(cov_matrix)  # Cholesky decomposition
        self.L_inv_ = jnp.linalg.inv(self.L_)      # Inverse of Cholesky factor

        return self

    def transform(self, X):
        """
        Apply the pipeline: log-transform (optional), scale, reduce dimensions, and prewhiten.
        Args:
            X (jnp.ndarray): Input data of shape (N, N_c).
        Returns:
            jnp.ndarray: Prewhitened data of shape (N, n_components).
        """
        # Step 1: Apply log-transformation (if enabled)
        if self.apply_log:
            X = jnp.log(X + self.shift_)

        # Step 2: Scale the data
        X_scaled = (X - self.means_) / self.stds_

        # Step 3: Apply PCA
        X_reduced = jnp.dot(X_scaled, self.components_.T)

        # Step 4: Prewhiten the data
        X_prewhitened = jnp.dot(X_reduced, self.L_inv_.T)
        return X_prewhitened

    def inverse_transform(self, X_prewhitened):
        """
        Reconstruct the original data from prewhitened data.
        Args:
            X_prewhitened (jnp.ndarray): Prewhitened data of shape (N, n_components).
        Returns:
            jnp.ndarray: Reconstructed original data of shape (N, N_c).
        """
        # Step 1: Reverse prewhitening
        X_reduced = jnp.dot(X_prewhitened, self.L_.T)

        # Step 2: Reverse PCA
        X_scaled = jnp.dot(X_reduced, self.components_)

        # Step 3: Reverse scaling
        X_original = X_scaled * self.stds_ + self.means_

        # Step 4: Reverse log-transformation (if enabled)
        if self.apply_log:
            X_original = jnp.exp(X_original) - self.shift_

        return X_original