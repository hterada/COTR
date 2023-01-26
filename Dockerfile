# for running the demo

FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

RUN apt-get update -y
RUN apt-get install -y wget git

# Miniconda
RUN mkdir -p install/miniconda
WORKDiR install/miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_22.11.1-1-Linux-x86_64.sh
RUN bash Miniconda3-py37_22.11.1-1-Linux-x86_64.sh -b
ENV PATH $PATH:/root/miniconda3/bin

# COTR env
RUN mkdir -p /COTR_env
WORKDiR /COTR_env
COPY environment.yml .
RUN conda env create -f environment.yml
RUN conda init bash
# replace shell
SHELL ["conda", "run", "-n", "cotr_env", "/bin/bash", "-c"]
RUN conda info -e
RUN conda install -c conda-forge glfw
RUN pip install torchprof
RUN conda install pandas

VOLUME ["/COTR"]
WORKDIR /COTR

# BUILD: $ docker build -t cotr/demo:1.0 .

# RUN: docker_run_demo.sh
