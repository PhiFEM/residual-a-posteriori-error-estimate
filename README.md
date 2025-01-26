# Phi-FEM with FEniCSx

A phi-FEM implementation for Poisson-Dirichlet problems.

## Prerequisites

- docker/podman

The docker image is based on FEniCSx.
It contains also some python libraries dependencies.

## Build the image:
Replace `YOUR_ENGINE_HERE` by `docker`, `podman` or your favorite container engine (with similar commands as Docker).
```bash
export CONTAINER_ENGINE="YOUR_ENGINE_HERE"
${CONTAINER_ENGINE} login
cd docker/
sudo -E bash build_image.sh
```

## Launch the image (from the root directory):
```bash
sudo -E bash run_image.sh
```

## Run an example (inside the container from the root directory):
Replace `TEST_CASE` by one of the available test cases:
```bash
cd demo/TEST_CASE/
```
The `main.py` scripts have the following parameters:
```bash
python3 main.py SOLVER SIZE_INIT_MESH NUM_REF_STEPS REFINEMENT_MODE (--exact_error)
```
where
- `SOLVER` defines the FE solver (`str` among `FEM` or `phiFEM`),
- `SIZE_INIT_MESH` defines the size of the initial mesh (`float`),
- `NUM_REF_STEPS` defines the number of refinement steps (`int`),
- `REFINEMENT_MODE` defines the refinement strategy to use (`str` among `uniform`, `H10` or `L2` --adaptive refinement based on the H10 or L2 error estimator--).
- `--exact_error` flag to compute a higher order approximation of the exact errors (requires to first run `main.py` with solver as `FEM`). Since this computation is based on solves of the FE problem on finer meshes and/or higher order spaces, this can drastically slow the computation and require more computational resources.

Example:
```bash
cd demo/pdt_sines/
python3 main.py phiFEM 0.1 25 H10
```

## Launch unit tests (inside the container from the root directory):
```bash
pytest
```
