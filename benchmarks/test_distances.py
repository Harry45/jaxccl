import pytest
import numpy as np
import pyccl as ccl
from pyccl.modified_gravity import MuSigmaMG

# from jax_cosmo
from jax_cosmo.core import Cosmology
import jax_cosmo.background as bg

np.set_printoptions(precision=4)

# Set tolerances
DISTANCES_TOLERANCE = 1e-4

# The distance tolerance is 1e-3 for distances with massive neutrinos
# This is because we compare to astropy which uses a fitting function
# instead of the full phasespace integral.
# The fitting function itself is not accurate to 1e-4.
DISTANCES_TOLERANCE_MNU = 1e-3

# Tolerance for comparison with CLASS. Currently works up to 1e-6.
DISTANCES_TOLERANCE_CLASS = 1e-5

# Set up the cosmological parameters to be used in each of the models
# Values that are the same for all 5 models
Omega_c = 0.25
Omega_b = 0.05
h = 0.7
A_s = 2.1e-9
n_s = 0.96
Neff = 0.

# jax_cosmo uses sigma8 as input
sigma8 = 0.8

# Introduce non-zero values of mu_0 and sigma_0 (mu / Sigma
# parameterisation of modified gravity) for some of the tests
mu_0 = 0.1
sigma_0 = 0.1

# Values that are different for the different models
Omega_v_vals = np.array([0.7, 0.7, 0.7, 0.65, 0.75])
w0_vals = np.array([-1.0, -0.9, -0.9, -0.9, -0.9])
wa_vals = np.array([0.0, 0.0, 0.1, 0.1, 0.1])

mnu = [[0.04, 0., 0.],
       [0.05, 0.01, 0.],
       [0.05, 0., 0.],
       [0.03, 0.02, 0.]]

# For tests with massive neutrinos, we require N_nu_rel + N_nu_mass = 3
# Because we compare with astropy for benchmarks
# Which assumes N total is split equally among all neutrinos.
Neff_mnu = 3.0


def read_chi_benchmark_file():
    """
    Read the file containing all the radial comoving distance benchmarks
    (distances are in Mpc/h)
    """
    # Load data from file
    dat = np.genfromtxt("./benchmarks/data/chi_model1-5.txt").T
    assert (dat.shape == (6, 6))

    # Split into redshift column and chi(z) columns
    z = dat[0]
    chi = dat[1:]
    return z, chi

def read_dm_benchmark_file():
    """
    Read the file containing all the distance modulus benchmarks
    """
    # Load data from file
    dat = np.genfromtxt("./benchmarks/data/dm_model1-5.txt").T
    assert (dat.shape == (6, 6))

    # Split into redshift column and chi(z) columns
    z = dat[0]
    dm = dat[1:]
    return z, dm


# Set-up test data
z, chi = read_chi_benchmark_file()
_, dm = read_dm_benchmark_file()

def compare_distances(z, chi_bench, dm_bench, Omega_v, w0, wa):
    """
    Compare distances calculated by pyccl with the distances in the benchmark
    file.
    This test is only valid when radiation is explicitly set to 0.
    """
    # Set Omega_K in a consistent way
    Omega_k = 1.0 - Omega_c - Omega_b - Omega_v

    cosmo = ccl.Cosmology(
        Omega_c=Omega_c, Omega_b=Omega_b, Neff=Neff,
        h=h, A_s=A_s, n_s=n_s, Omega_k=Omega_k,
        w0=w0, wa=wa, Omega_g=0)

    cosmo_jax = Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, Neff=Neff,
        h=h, sigma8=sigma8, n_s=n_s, Omega_k=Omega_k,
        w0=w0, wa=wa, Omega_g=0)


    # Calculate distance using pyccl
    a = 1. / (1. + z)
    chi = ccl.comoving_radial_distance(cosmo, a) * h
    chi_jccl = bg.radial_comoving_distance(cosmo_jax, a)

    # Compare to benchmark data
    assert np.allclose(chi, chi_bench, atol=1e-12, rtol=DISTANCES_TOLERANCE)
    assert np.allclose(chi_jccl, chi_bench, atol=1e-12, rtol=DISTANCES_TOLERANCE)

    # compare distance moudli where a!=1
    a_not_one = (a != 1).nonzero()
    dm = ccl.distance_modulus(cosmo, a[a_not_one])
    dm_jccl = bg.distance_modulus(cosmo_jax, a[a_not_one])

    assert np.allclose(dm, dm_bench[a_not_one], atol=1e-3, rtol=DISTANCES_TOLERANCE*10)
    assert np.allclose(dm_jccl, dm_bench[a_not_one], atol=1e-3, rtol=DISTANCES_TOLERANCE*10)

@pytest.mark.parametrize('i', list(range(5)))
def test_distance_model(i):
    compare_distances(z, chi[i], dm[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])