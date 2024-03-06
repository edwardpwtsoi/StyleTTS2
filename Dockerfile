FROM nvcr.io/nvidia/pytorch:23.10-py3

LABEL authors="edward.tsoi"

WORKDIR /home/appuser

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

RUN apt-get update
RUN apt-get install -y espeak-ng git-lfs

RUN git lfs install
RUN git clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS

RUN git clone https://github.com/edwardpwtsoi/StyleTTS2
RUN pip install -r StyleTTS2/requirements.txt

RUN mkdir -p StyleTTS2/Models/LibriTTS
RUN mv StyleTTS2-LibriTTS/Models/LibriTTS StyleTTS2/Models/LibriTTS
RUN rm -r StyleTTS2-LibriTTS

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN chown -R appuser /home/appuser
RUN chmod -R 775 /home/appuser
USER appuser

WORKDIR /home/appuser/StyleTTS2

ENTRYPOINT ["python", "train_finetune.py", "--config_path"]