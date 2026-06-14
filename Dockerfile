FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.3.1+cpu torchvision==0.18.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY api/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN python -c "import easyocr; easyocr.Reader(['es', 'en'], gpu=False, verbose=False)"

COPY api/ ./api/
COPY checkpoints/best_densenet_v2.pth ./checkpoints/best_densenet_v2.pth
COPY Notebooks/ta005f_final_results.json ./Notebooks/ta005f_final_results.json
COPY Notebooks/best_params_densenet.json ./Notebooks/best_params_densenet.json

WORKDIR /app/api

EXPOSE 8000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
