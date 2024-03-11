FROM nvcr.io/nvidia/pytorch:23.10-py3

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

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN sudo ./aws/install

RUN git lfs install
RUN git clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS

RUN git clone https://github.com/edwardpwtsoi/StyleTTS2
RUN python -m pip install -r StyleTTS2/requirements.txt

RUN mkdir -p StyleTTS2/Models/LibriTTS
RUN mv StyleTTS2-LibriTTS/Models/LibriTTS StyleTTS2/Models/LibriTTS
RUN rm -r StyleTTS2-LibriTTS

WORKDIR /workspace/StyleTTS2
