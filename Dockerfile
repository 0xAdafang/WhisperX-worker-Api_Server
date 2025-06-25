# Base image avec CUDA
FROM runpod/base:0.6.2-cuda12.4.1


RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git wget ffmpeg curl libsndfile1


RUN mkdir -p /cache/models
RUN mkdir -p /root/.cache/torch


COPY builder/requirements.txt /builder/requirements.txt


RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /builder/requirements.txt


COPY models/whisperx-vad-segmentation.bin /root/.cache/huggingface/hub/models--m-bain--whisperx/snapshots/last/whisperx-vad-segmentation.bin


COPY app /app
WORKDIR /app


COPY builder/download_models.sh /builder/download_models.sh
RUN chmod +x /builder/download_models.sh

# Exposition API
EXPOSE 8000

# Lancement de l’API FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
