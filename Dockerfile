# Stage 1: Build the virtual environment
FROM python:3.11-slim as builder

# Install uv
RUN pip install uv

# Create a virtual environment
RUN python -m venv /opt/venv

# Activate the virtual environment and install dependencies using uv
COPY pyproject.toml .
RUN . /opt/venv/bin/activate && uv pip install --no-cache-dir -r pyproject.toml

# Stage 2: Create the final image
FROM python:3.11-slim

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Add the virtual environment to the PATH
ENV PATH="/opt/venv/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the application code
COPY src/ .

# Set the entrypoint
ENTRYPOINT ["python", "-m", "crypto_analysis.run_pipeline"]
