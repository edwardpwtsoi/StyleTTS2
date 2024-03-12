FROM nvcr.io/nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

LABEL authors="edward.tsoi"

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	ca-certificates python3 python3-dev python3-pip git git-lfs curl unzip wget sudo ninja-build espeak-ng
WORKDIR /workspace

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip -qq awscliv2.zip
RUN sudo ./aws/install

RUN git lfs install
RUN git clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS

RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/edwardpwtsoi/StyleTTS2
RUN pip install -r StyleTTS2/requirements.txt

RUN mkdir -p StyleTTS2/Models/LibriTTS
RUN mv StyleTTS2-LibriTTS/Models/LibriTTS StyleTTS2/Models
RUN rm -r StyleTTS2-LibriTTS

WORKDIR /workspace/StyleTTS2
