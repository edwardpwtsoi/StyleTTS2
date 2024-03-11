ARG PYTHON_VERSION=3.10

FROM nvcr.io/nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

LABEL authors="edward.tsoi"

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
	ca-certificates \
    git \
    git-lfs \
    curl \
    unzip \
    wget \
    sudo \
    espeak-ng

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN chown -R appuser /home/appuser
RUN chmod -R 775 /home/appuser
USER appuser
WORKDIR /home/appuser

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN sudo ./aws/install

RUN git lfs install
RUN git clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS

RUN curl -fsSL -v -o ~/miniconda.sh -O  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
# Manually invoke bash on miniconda script per https://github.com/conda/conda/issues/10431
RUN chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /home/appuser/.local/conda && \
    rm ~/miniconda.sh && \
    /home/appuser/.local/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython && \
    /home/appuser/.local/conda clean -ya


RUN /home/appuser/.local/conda/bin/python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/edwardpwtsoi/StyleTTS2
RUN /home/appuser/.local/conda/bin/python -m pip install -r StyleTTS2/requirements.txt

RUN mkdir -p StyleTTS2/Models/LibriTTS
RUN mv StyleTTS2-LibriTTS/Models/LibriTTS StyleTTS2/Models/LibriTTS
RUN rm -r StyleTTS2-LibriTTS

ENV PATH /home/appuser/.local/conda/conda/bin:$PATH

WORKDIR /home/appuser/StyleTTS2
