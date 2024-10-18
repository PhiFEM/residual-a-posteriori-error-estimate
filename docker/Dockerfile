FROM dolfinx/dolfinx:stable
# RUN apt-get -y update && apt-get install -y python3-venv

# Path to venv
# RUN mkdir /opt/venv
# ENV VIRTUAL_ENV=/opt/venv

# Create venv
# RUN python3 -m venv --system-site-packages $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies:
RUN pip install matplotlib pandas jax
ENV PYTHONPATH="$PYTHONPATH:/home/dolfinx/shared/phiFEM/"