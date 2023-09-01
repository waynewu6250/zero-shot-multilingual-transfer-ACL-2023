# Install pytorch
python3.8 -m pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch

# Install datasets
conda install -c huggingface -c conda-forge datasets

# Install other packages
python3.8 -m pip install -r requirements.txt

# Install adapter-transformers
git clone https://github.com/adapter-hub/adapter-transformers.git; cd adapter-transformers; python3.8 -m pip install -e .