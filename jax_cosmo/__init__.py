# # Cosmology in JAX
# from pkg_resources import DistributionNotFound
# from pkg_resources import get_distribution

# try:
#     __version__ = get_distribution(__name__).version
# except DistributionNotFound:
#     # package is not installed
#     pass

import jax_cosmo.angular_cl as cl
import jax_cosmo.background as background
import jax_cosmo.bias as bias
import jax_cosmo.likelihood as likelihood
import jax_cosmo.power as power
import jax_cosmo.probes as probes
import jax_cosmo.redshift as redshift
import jax_cosmo.transfer as transfer
import jax_cosmo.core as jcore
import jax_cosmo.parameters as jparam
import jax_cosmo.sparse as sparse
