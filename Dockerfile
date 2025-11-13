FROM quay.io/pypa/manylinux2014_x86_64

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install matching version of Python to cluster (3.12.3)

# Install Python dependencies
RUN pip3 install maturin

# Copy project files
WORKDIR /app
COPY . .

# Build wheel
RUN maturin build --release --manylinux 2014

# The wheel will be in /app/target/wheels/
