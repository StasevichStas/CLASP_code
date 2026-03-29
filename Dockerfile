FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /workspace/datasets/coco/images /workspace/datasets/coco/annotations.json /workspace/result_img /workspace/.cache/torch

CMD ["python", "clasp.py"]