# for running the demo

FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

RUN apt-get update -y
RUN apt-get install -y wget

# Miniconda
RUN mkdir -p install/miniconda
WORKDiR install/miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_22.11.1-1-Linux-x86_64.sh
RUN bash Miniconda3-py37_22.11.1-1-Linux-x86_64.sh -b
ENV PATH $PATH:/root/miniconda3/bin

# COTR env
RUN mkdir -p /COTR
WORKDiR /COTR
COPY environment.yml .
RUN conda env create -f environment.yml
RUN conda init bash
# replace shell
SHELL ["conda", "run", "-n", "cotr_env", "/bin/bash", "-c"]
RUN conda info -e

## Download the pretrained weights
RUN wget -c https://www.cs.ubc.ca/research/kmyi_data/files/2021/cotr/default.zip
RUN mkdir -p out
RUN apt-get install -y zip
RUN unzip -o -d out default.zip



