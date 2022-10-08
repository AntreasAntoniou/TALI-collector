FROM gcr.io/deeplearning-platform-release/pytorch-cpu.1-4:latest

SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y \
    git \
    wget \
    unzip \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    fish

RUN conda init bash
RUN conda create -n tali-collector python=3.10 -y
RUN echo "conda activate tali-collector" >> ~/.bashrc

RUN git clone https://github.com/AntreasAntoniou/TALI-collector.git

SHELL ["conda", "run", "-n", "tali-collector", "/bin/bash", "-c"]

RUN cd TALI-collector \
    && conda install -c conda-forge mamba -y \
    && mamba install pytorch torchvision torchaudio -c pytorch -c conda-forge \
    && bash install_dependencies.sh \
    && yes | pip install transformers -U


RUN cd TALI-collector \
    && git pull

WORKDIR /TALI-collector

ENTRYPOINT ["/bin/bash"]