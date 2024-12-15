FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/N01N9/WhisperLLM.git /app/WhisperLLM

RUN pip install --no-cache-dir -r /app/WhisperLLM/requirements.txt