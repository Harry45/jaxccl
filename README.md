# Differentiable CCL using JAX

As a first step, we are using the existing implementations of `jax_cosmo` (see Github repository [here](https://github.com/DifferentiableUniverseInitiative/jax_cosmo)). We are expanding on it to accommodate for more functionalities as implemented in CCL.

## Running tests

Within the `jaxccl/` folder:

```bash
PYTHONPATH=$(pwd) pytest -s benchmarks/test_distances.py
```

## Compiling the Documentation

```bash
mkdocs build
mkdocs serve
```

## Other Functionalities Included