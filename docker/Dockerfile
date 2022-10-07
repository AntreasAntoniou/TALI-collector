FROM gcr.io/tpu-pytorch/xla:nightly_3.8_tpuvm

SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN conda init bash
RUN conda create -n tali-collector python=3.10 -y
RUN echo "conda activate tali-collector" >> ~/.bashrc

RUN git clone https://github.com/AntreasAntoniou/TALI-collector.git

SHELL ["conda", "run", "-n", "tali-collector", "/bin/bash", "-c"]

RUN cd TALI-collector \
    && conda install -c conda-forge mamba -y \
    && mamba install pytorch torchvision torchaudio -c pytorch -c conda-forge \
    && bash install_dependencies.sh

RUN apt install fish -y \
    && mamba install starship -y

RUN cd TALI-collector \
    && git pull

WORKDIR /TALI-collector

CMD ["bin/bash"]