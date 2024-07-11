##########################################
#   RAGMT Dockerfile pytorch-gpu
##########################################

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main

WORKDIR /home

# If set to nothing, will install the latest version
ARG PYTORCH='2.1.1'
ARG TORCH_VISION=''
ARG TORCH_AUDIO=''
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu121'

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_VISION} -gt 0 ] && VERSION='torchvision=='TORCH_VISION'.*' ||  VERSION='torchvision'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_AUDIO} -gt 0 ] && VERSION='torchaudio=='TORCH_AUDIO'.*' ||  VERSION='torchaudio'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA



COPY transformers /home/transformers
RUN cd transformers && git checkout $REF


RUN python3 -m pip install --no-cache-dir -e ./transformers[torch]

RUN python3 -m pip uninstall -y tensorflow flax

RUN python3 -m pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git pytesseract
RUN python3 -m pip install -U "itsdangerous<2.1.0"

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

RUN apt install git python3-git python-is-python3 -y
RUN python3 -m pip uninstall torch torchvision torchaudio -y
RUN python3 -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN python3 -m pip install -r /home/transformers/examples/research_projects/rag-end2end-retriever/requirements.txt
RUN git config --global --add safe.directory /home/transformers
RUN pip install scipy scikit-learn nltk rouge_score
RUN pip install -U jupyter ipywidgets
RUN pip install streamlit
RUN pip install spacy
RUN pip install googlesearch-python
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download de_core_news_sm
RUN pip install deepspeed
RUN pip install faiss-gpu
RUN pip install sacrebleu sacremoses sentencepiece langdetect tensorboard tensorboardX wandb
RUN pip install lightning

CMD "/bin/bash"