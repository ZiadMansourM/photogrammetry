# Base image
FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libc-dev \
    gcc \
    libx11-6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install tensorflow
RUN pip install opencv-python
RUN pip install scikit-learn
RUN pip install Pympler
RUN pip install rembg
RUN pip install hdbscan
RUN pip install open3d

# Copy the app code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]