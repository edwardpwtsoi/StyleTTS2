FROM nvcr.io/nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

LABEL authors="edward.tsoi"

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	ca-certificates python3-dev git curl unzip wget sudo ninja-build espeak-ng git-lfs
RUN ln -sv /usr/bin/python3 /usr/bin/python

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

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py \

RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/edwardpwtsoi/StyleTTS2
RUN pip install -r StyleTTS2/requirements.txt

RUN mkdir -p StyleTTS2/Models/LibriTTS
RUN mv StyleTTS2-LibriTTS/Models/LibriTTS StyleTTS2/Models/LibriTTS
RUN rm -r StyleTTS2-LibriTTS

WORKDIR /home/appuser/StyleTTS2
