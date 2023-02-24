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
# SHELL ["conda", "run", "-n", "cotr_env", "/bin/bash", "-c"]
# RUN conda info -e
RUN conda install -c conda-forge -n cotr_env glfw -y
RUN conda run -n cotr_env /bin/bash -c "pip install torchprof"
RUN conda install -n cotr_env pandas -y
RUN conda update -n cotr_env --all -y
RUN conda install pytorch torchvision==0.13.1 torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -n cotr_env -y
RUN conda clean --all -y
RUN conda run -n cotr_env pip install pytorch_memlab
RUN conda install -n cotr_env jupyterlab ipywidgets -y
RUN conda install -n cotr_env -c conda-forge torchinfo tensorboardx -y
RUN conda run -n cotr_env pip install jupyterlab_tabnine
RUN conda install -c conda-forge line_profiler -y


VOLUME ["/COTR"]
WORKDIR /COTR

# BUILD: $ docker build -t cotr/demo:1.0 .

# RUN: docker_run_demo.sh
