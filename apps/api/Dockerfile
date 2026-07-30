# ──────────────────────────────────────────────────────────────
# Sahayak API — Dockerfile (the "recipe" to build our backend image)
# ──────────────────────────────────────────────────────────────
# A Docker image is a sealed, reproducible package: Python + our code +
# our dependencies. Anyone running `docker compose up` gets the EXACT
# same environment — no "works on my machine" surprises.

# Start FROM an official Python image (slim = smaller, no bloat).
FROM python:3.12-slim

# This prevents Python from buffering logs, so we see output live in the terminal.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container (like `cd /app`).
WORKDIR /app

# Install dependencies FIRST (this layer is cached — if requirements.txt
# doesn't change, Docker reuses it, making rebuilds fast).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy our actual application code into the container.
COPY . .

# Expose the port the API listens on (documentation only; compose handles mapping).
EXPOSE 8000

# Run the app with uvicorn. `--host 0.0.0.0` = listen on all interfaces
# (needed inside Docker so other containers / your browser can reach it).
# `--reload` makes the server auto-restart when code changes (dev convenience).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
