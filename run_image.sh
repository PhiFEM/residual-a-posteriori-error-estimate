#!/bin/bash
podman run --privileged -ti -p 127.0.0.1:8080:8080 -v $(pwd):/home/dolfinx/shared -w /home/dolfinx/shared rbulle/phifemx:latest