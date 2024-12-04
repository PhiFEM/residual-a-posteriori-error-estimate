FROM dolfinx/dolfinx:stable
RUN python3 -m pip install -U pip
# RUN apt-get -y update && apt-get install -y python3-venv

# Install mmg2d
# ENV BIN_DIR="/usr/local/bin/"
# WORKDIR "$BIN_DIR"
# RUN git clone https://github.com/MmgTools/mmg.git
# WORKDIR "$BIN_DIR/mmg"
# RUN mkdir build
# WORKDIR "$BIN_DIR/mmg/build"
# RUN cmake ..
# RUN make 
# ENV PATH="$PATH:/usr/local/bin/mmg/build/bin"
# RUN git clone https://github.com/alistairboyer/MarchingNumPy.git
# WORKDIR "$BIN_DIR/MarchingNumPy"
# RUN pip3 install .
# ENV PYTHONPATH="$PYTHONPATH:/usr/local/bin/MarchingNumPy/"

# Install dependencies:
RUN pip install pandas jax meshio pygmsh h5py lxml
ENV PYTHONPATH="$PYTHONPATH:/home/dolfinx/shared/phiFEM/"