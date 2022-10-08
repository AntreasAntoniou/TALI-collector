FROM condaforge/mambaforge:latest

SHELL ["/bin/bash", "-c"]

RUN apt update
RUN apt install fish -y

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

RUN pwd
ADD entrypoint.sh /entrypoint.sh

RUN echo "starship init fish | source" >> ~/.config/fish/config.fish
RUN  echo 'eval "$(starship init bash)"' >> ~/.bashrc

WORKDIR /TALI-collector

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]

# The Dockerfile is pretty straightforward. It starts with the base image,
# which is the mambaforge image. Then, it installs fish and creates a conda environment.
# It clones the TALI-collector repository and installs the dependencies.
# Finally, it sets the working directory to the TALI-collector repository and sets the
# entrypoint to the entrypoint.sh script.