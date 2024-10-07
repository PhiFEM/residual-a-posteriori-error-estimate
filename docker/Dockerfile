FROM dolfinx/dev-env:stable
RUN apt-get -y update && apt-get install -y python3-venv

# Path to venv
RUN mkdir /opt/venv
ENV VIRTUAL_ENV=/opt/venv

# Allow user fenics to access venv directories
# RUN chown fenics: $VIRTUAL_ENV

# Create venv
RUN python3 -m venv --system-site-packages $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies:
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -IUr /app/requirements.txt