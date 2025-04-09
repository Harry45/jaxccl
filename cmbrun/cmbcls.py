import numpy as np
from classy import Class
from typing import Tuple, List, Dict, Union
import matplotlib.pylab as plt
from ml_collections.config_dict import ConfigDict


def get_config(experiment) -> ConfigDict:
    """Generates and returns a configuration dictionary for the emulator.

    This function sets up various configuration parameters for the emulator,
    including neutrino settings, CLASS settings, priors, and emulator settings.

    Returns:
        ConfigDict: A configuration dictionary containing the following keys:
            - neutrino: Neutrino settings.
            - classy: CLASS settings.
            - priors: Prior distributions for various parameters.
            - emu: Emulator settings.
    """
    config = ConfigDict()
    config.logname = "cmb_power_spectra"
    config.experiment = experiment

    # cosmological parameters
    config.cosmo = cosmo = ConfigDict()
    cosmo.names = ["sigma8", "Omega_cdm", "Omega_b", "h", "n_s"]

    # neutrino settings
    config.neutrino = neutrino = ConfigDict()
    neutrino.N_ncdm = 1.0
    neutrino.deg_ncdm = 3.0
    neutrino.T_ncdm = 0.71611
    neutrino.N_ur = 0.00641
    neutrino.fixed_nm = 0.06

    # CLASS settings
    config.classy = classy = ConfigDict()
    classy.output = "tCl,pCl,lCl,mPk"
    classy.Omega_k = 0.0
    classy.k_max_pk = 50

    # priors
    config.priors = {
        "sigma8": {
            "distribution": "uniform",
            "loc": 0.7,
            "scale": 0.2,
            "fiducial": 0.8,
        },
        "Omega_cdm": {
            "distribution": "uniform",
            "loc": 0.20,
            "scale": 0.15,
            "fiducial": 0.2,
        },
        "Omega_b": {
            "distribution": "uniform",
            "loc": 0.04,
            "scale": 0.02,
            "fiducial": 0.045,
        },
        "h": {"distribution": "uniform", "loc": 0.62, "scale": 0.12, "fiducial": 0.68},
        "n_s": {"distribution": "uniform", "loc": 0.90, "scale": 0.2, "fiducial": 1.0},
    }

    # emulator settings
    config.emu = emu = ConfigDict()
    emu.nlhs = 1000
    emu.jitter = 1e-10
    emu.lr = 0.01
    emu.nrestart = 1
    emu.niter = 500

    return config


def class_args(config: ConfigDict) -> Dict:
    """Generates CLASS arguments to be passed to classy to compute the different quantities.
    Args:
        config (ConfigDict): A configuration file containing the parameters.
    Returns:
        dict: A dictionary to input to class
    """
    dictionary = dict()
    dictionary["output"] = config.classy.output
    dictionary["P_k_max_1/Mpc"] = config.classy.k_max_pk
    dictionary["Omega_k"] = config.classy.Omega_k
    return dictionary


def neutrino_args(config: ConfigDict) -> Dict:
    """Generates a dictionary for the neutrino settings.
    Args:
        config (ConfigDict): The main configuration file
    Returns:
        dict: A dictionary with the neutrino parameters.
    """
    dictionary = dict()
    dictionary["N_ncdm"] = config.neutrino.N_ncdm
    dictionary["deg_ncdm"] = config.neutrino.deg_ncdm
    dictionary["T_ncdm"] = config.neutrino.T_ncdm
    dictionary["N_ur"] = config.neutrino.N_ur
    dictionary["m_ncdm"] = config.neutrino.fixed_nm / config.neutrino.deg_ncdm
    return dictionary


def delete_module(class_module: Class):
    """Deletes the module to prevent memory overflow.
    Args:
        module (Class): A CLASS module
    """
    class_module.struct_cleanup()

    class_module.empty()

    del class_module


def class_compute(config: ConfigDict, cosmology: Dict) -> Class:
    """Pre-computes the quantities in CLASS.
    Args:
        config (ConfigDict): The main configuration file for running Class
        cosmology (dict): A dictionary with the cosmological parameters
    Returns:
        Class: A CLASS module
    """
    # generates the dictionaries to input to Class
    arg_class = class_args(config)
    arg_neutrino = neutrino_args(config)

    # Run Class
    class_module = Class()
    class_module.set(arg_class)
    class_module.set(arg_neutrino)
    class_module.set(cosmology)
    class_module.compute()

    return class_module


def calculate_cmb_cls_class(
    cosmology: Dict, cfg: ConfigDict, ellmax: int = 2500
) -> np.ndarray:
    """Calculate the linear matter power spectrum using CLASS.

    Args:
        cosmology (Dict): A dictionary containing cosmological parameters.
        cfg (ConfigDict): A configuration dictionary containing CLASS settings.

    Returns:
        np.ndarray: An array of linear matter power spectrum values.

    """
    cmodule = class_compute(cfg, cosmology)
    cells = cmodule.raw_cl(ellmax)
    factor = 2.7255e6**2 * cells["ell"] * (cells["ell"] + 1) / (2 * np.pi)
    cls_tt = cells["tt"] * factor
    cls_ee = cells["ee"] * factor
    cls_te = cells["te"] * factor
    delete_module(cmodule)
    return cells["ell"][2:], cls_tt[2:], cls_ee[2:], cls_te[2:]


def compute_cmb_gradients(cosmology: np.ndarray, cfg, eps=1e-2):
    """Computes the first derivative of each CMB power spectrum (TT, EE, TE)
    with respect to each cosmological parameter using finite difference.

    Args:
        cosmology (jnp.ndarray): Array of cosmological parameters.
        cfg: Configuration object containing parameter names and settings.
        eps (float): Finite difference step size.

    Returns:
        jnp.ndarray: A 3D array of shape (num_ells, num_params, 3), where:
            - num_ells is the number of multipoles,
            - num_params is the number of cosmological parameters,
            - The last dimension corresponds to TT, EE, and TE derivatives.
    """
    num_params = len(cfg.cosmo.names)
    num_ells = 2500 - 1

    # Initialize gradient arrays
    gradients = np.zeros((num_ells, num_params, 3))  # TT, EE, TE derivatives

    # Compute fiducial power spectra
    cosmo_dict = {name: cosmology[i] for i, name in enumerate(cfg.cosmo.names)}
    _, cls_tt_fid, cls_ee_fid, cls_te_fid = calculate_cmb_cls_class(cosmo_dict, cfg)

    for i, name in enumerate(cfg.cosmo.names):
        # Perturb the cosmological parameter
        cosmo_dict_perturbed = cosmo_dict.copy()
        delta = eps * cosmology[i]
        cosmo_dict_perturbed[name] += delta  # Forward step

        # Compute perturbed power spectra
        _, cls_tt_pert, cls_ee_pert, cls_te_pert = calculate_cmb_cls_class(
            cosmo_dict_perturbed, cfg
        )

        # Compute finite difference derivative
        gradients[:, i, 0] = (cls_tt_pert - cls_tt_fid) / delta  # dC_ell^TT / dparam
        gradients[:, i, 1] = (cls_ee_pert - cls_ee_fid) / delta  # dC_ell^EE / dparam
        gradients[:, i, 2] = (cls_te_pert - cls_te_fid) / delta  # dC_ell^TE / dparam

    grad = {
        "tt": gradients[:, :, 0],
        "ee": gradients[:, :, 1],
        "te": gradients[:, :, 2],
    }
    return grad
