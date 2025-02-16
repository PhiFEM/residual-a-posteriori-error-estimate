# Residual-based a posteriori error estimates with boundary correction for $\varphi$-FEM

This repository contains the code used to generate the numerical results in the publication:

*Residual-based a posteriori error estimates with boundary correction for $\varphi$-FEM*  
R. Becker, R. Bulle, M. Duprez, V. Lleras  
[https://hal.science/hal-04931977](https://hal.science/hal-04931977)  

## This repository is for reproducibility purposes only
It is "frozen in time" and not maintained.
To use our $\varphi$-FEM code please refer to the [phiFEM repository](https://github.com/PhiFEM/Poisson-Dirichlet-fenicsx).

## Generate the results

### Prerequisites

- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/)/[podman](https://podman.io/)

The image is based on [FEniCSx](https://fenicsproject.org/).
It contains also some python libraries dependencies.

### Install and launch the residual-phifem container

Replace `YOUR_ENGINE` by `docker`, `podman` or your favorite container engine (the following instructions are using Docker/podman UI).

1) Clone this repository in a dedicated directory:
   
   ```bash
   mkdir residual-phifem/
   git clone https://github.com/PhiFEM/residual-a-posteriori-error-estimate.git residual-phifem
   ```

2) Download the image from the docker.io registry, in the main directory:
   
   ```bash
   export CONTAINER_ENGINE=YOUR_ENGINE
   cd residual-phifem
   bash pull_image.sh
   ```

3) Launch the container:

   ```bash
   bash run_image.sh
   ```

### Generate the publication results

Inside the container:
   
   ```bash
   cd demo/
   python3 generate_results.py
   ```

### Find the results

The numerical results generated by `generate_results.py` are organized as follows:

```
demo
├── flower
│   └── output_phiFEM       # Contains the phiFEM solver results
│       ├── H10             # Contains the adaptive refinement loop results
│       │   ├── functions   # Contains many functions esp. eta and its components, u_h and w_h
│       │   ├── meshes      # Contains the meshes generated by the refinement loop
│       │   ├── results.csv # Contains the convergence data
│       │   └── slopes.csv  # Contains the convergence slopes
│       └── uniform         # Contains the uniform refinement loop results
│           ├── functions   
│           ├── meshes      
│           ├── results.csv
│           └── slopes.csv
├── lshaped
│   ├── output_FEM          # Contains the FEM solver results
│   │   ├── H10
│   │   │   ├── functions
│   │   │   ├── meshes
│   │   │   ├── results.csv
│   │   │   └── slopes.csv
│   │   └── uniform
│   │       ├── functions   
│   │       ├── meshes      
│   │       ├── results.csv
│   │       └── slopes.csv
│   ├── output_FEM
│   │   ├── H10
│   │   │   ├── functions   
│   │   │   ├── meshes      
│   │   │   ├── results.csv
│   │   │   └── slopes.csv
│   │   └── uniform
│   │       ├── functions   
│   │       ├── meshes      
│   │       ├── results.csv
│   │       └── slopes.csv
│   └── output_phiFEM
│       ├── H10
│       │   ├── functions   
│       │   ├── meshes      
│       │   ├── results.csv
│       │   └── slopes.csv
│       └── uniform
│           ├── functions   
│           ├── meshes      
│           ├── results.csv
│           └── slopes.csv
└── pdt_sines_pyramid
    ├── output_FEM
    │   ├── H10
    │   │   ├── functions   
    │   │   ├── meshes      
    │   │   ├── results.csv
    │   │   └── slopes.csv
    │   ├── uniform
    │   │   ├── functions   
    │   │   ├── meshes      
    │   │   ├── results.csv
    │   │   └── slopes.csv
    └── output_phiFEM
        ├── H10
        │   ├── functions   
        │   ├── meshes      
        │   ├── results.csv
        │   └── slopes.csv
        └── uniform
            ├── functions   
            ├── meshes      
            ├── results.csv
            └── slopes.csv
```

## Compute a specific case individually:

It is also possible to run a single demo with custom parameters with

```bash
cd demo/
python3 main.py DEMO_NAME SOLVER_NAME
```
where `DEMO_NAME` is one of `lshaped, flower, pdt_sines_pyramid` and `SOLVER_NAME` one of `FEM, phiFEM`.

Other parameters can be modified in the `.yaml` files:

```bash
demo
└── DEMO_NAME
    └── parameters.yaml
```

> **Remark:** `python3 main.py flower FEM` does not work due to the absence of refinement strategy for curved boundaries.

## Issues and support

Please use the issue tracker to report any issues.

## Authors (alphabetical)

Roland Becker, Université de Pau et des Pays de l'Adour  
Raphaël Bulle, Inria Nancy Grand-Est  
Michel Duprez, Inria Nancy Grand-Est  
Vanessa Lleras, Université de Montpellier

## License

residual-phifem is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with residual-phifem. If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).