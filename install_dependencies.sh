# -------------------------------------------
# This script installs all the dependencies need to use tali-collector
mamba install numpy transformers tqdm rich pytube datasets Pillow -y
mamba install pytorch torchvision torchaudio -c pytorch -c conda-forge -y
yes | pip install yelp_uri clip

# This script installs all the dependencies need to use dev on tali-collector
# --------------------------------------------
mamba install -c conda-forge jupyterlab black isort -y