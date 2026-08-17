FROM python:3.10-slim

# ZeroGPU Spaces use a base image provided by Hugging Face.
# This Dockerfile is for reference — the actual Space uses
# the HF-managed Gradio SDK container.
#
# If you need custom dependencies not in requirements.txt,
# uncomment and modify the following lines:

# RUN apt-get update && apt-get install -y \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY app.py .

# EXPOSE 7860
# ENV GRADIO_SERVER_NAME=0.0.0.0
# ENV GRADIO_SERVER_PORT=7860

# CMD ["python", "app.py"]