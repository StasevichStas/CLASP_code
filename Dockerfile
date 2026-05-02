# Используем devel-образ, чтобы была возможность компилировать C++ расширения (для pydensecrf и подобных)
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# Добавь это в блок системных зависимостей (пункт 1)
RUN apt-get update && apt-get install -y openssh-server && mkdir /var/run/sshd

# Это позволит Clore подкидывать твой ключ в нужную папку
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# 1. Системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# 2. КРИТИЧЕСКИЙ ШАГ: Обновление PyTorch для RTX 5090 (sm_120 / Blackwell)
# Мы удаляем стандартный torch и ставим Nightly-сборку с поддержкой CUDA 12.8/13.0.
# Это единственный способ убрать ошибку "no kernel image is available".
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Установка зависимостей из requirements.txt
COPY requirements.txt .
# Очищаем requirements от пакетов nvidia-*, чтобы они не конфликтовали с системной CUDA
RUN grep -v "nvidia-" requirements.txt > req_clean.txt && \
    pip install --no-cache-dir -r req_clean.txt

# 4. Установка библиотек, требующих компиляции (pydensecrf и другие)
RUN pip install --no-cache-dir git+https://github.com/lucasb-eyer/pydensecrf.git huggingface_hub accelerate

# 5. Предварительная загрузка весов моделей (чтобы не качать их при каждом запуске)
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

RUN python3 -c "import timm; \
timm.create_model('vit_base_patch14_dinov2', pretrained=True); \
timm.create_model('vit_large_patch14_dinov2', pretrained=True)"

RUN python3 -c "from transformers import pipeline; \
pipeline(model='facebook/dinov3-vits16-pretrain-lvd1689m', task='image-feature-extraction'); \
pipeline(model='facebook/dinov3-vitb16-pretrain-lvd1689m', task='image-feature-extraction'); \
pipeline(model='facebook/dinov3-vitl16-pretrain-lvd1689m', task='image-feature-extraction')"

# 6. Копируем исходный код проекта
COPY . .

# Создаем папки для результатов, если их нет
RUN mkdir -p result_img examples

# 7. Запуск скрипта
CMD service ssh start && python clasp.py && sleep infinity