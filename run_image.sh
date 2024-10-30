#!/bin/bash
podman run --privileged -ti -v $(pwd):/home/dolfinx/shared -w /home/dolfinx/shared rbulle/phifemx:latest