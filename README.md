# Phi-FEM with FEniCSx

The docker image is based on FEniCSx.
It contains also some python libraries dependencies.

To build the image:
```bash
cd docker
sudo bash build_image.sh
```

To launch the image (from the root directory):
```bash
sudo bash run_image.sh
```

<!--- Run an example (from the root directory):
```bash
cd Poisson-Dirichlet
python3 phiFEM_test_case1.py
``` --->