# Use an official Python runtime as a parent image.
FROM docker.io/library/python:3.10-slim-bullseye

# Set the working directory in the container.
WORKDIR /lit

# Install system dependencies required by PyTorch and others.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements.txt file into the container and install dependencies.
COPY requirements.txt .

# Update pip and setuptools before installing other dependencies to prevent errors
RUN pip install --upgrade pip setuptools

RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire application code (all the files & directories where the Dockerfile is), including the model weights, into the container.
# This assumes your model is at ./app/models/Spam-Classifier-GPT2-Model.pt relative to the project hugging face root.
# Another example, copies ./app/main.py
COPY . /lit

# Expose the port that Streamlit runs on.
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the Streamlit application when the container starts. (starting from /lit)
ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false", "--server.fileWatcherType=none"]