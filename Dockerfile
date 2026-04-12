# FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install git+https://github.com/lucasb-eyer/pydensecrf.git huggingface_hub accelerate

# RUN python3 -c "import timm; timm.create_model('vit_base_patch14_dinov2', pretrained=True)"
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

RUN python3 -c "import timm; \
timm.create_model('vit_small_patch14_dinov2', pretrained=True); \
timm.create_model('vit_base_patch14_dinov2', pretrained=True); \
timm.create_model('vit_large_patch14_dinov2', pretrained=True)"

RUN python3 -c "from transformers import pipeline; \
pipeline(model='facebook/dinov3-vits16-pretrain-lvd1689m', task='image-feature-extraction'); \
pipeline(model='facebook/dinov3-vitb16-pretrain-lvd1689m', task='image-feature-extraction'); \
pipeline(model='facebook/dinov3-vitl16-pretrain-lvd1689m', task='image-feature-extraction')"

COPY . .

RUN mkdir -p result_img examples

CMD ["python", "clasp.py"]