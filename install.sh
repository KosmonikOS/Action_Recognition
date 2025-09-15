# Shows how to install mmpose on MacOS
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision cpuonly -c pytorch
pip install -U openmim

# Fix error with xtcocotools
xcode-select --install || true
conda activate act-rec
python -m pip install -U "pip<24.3" "setuptools<70" "wheel>=0.41"
python -m pip install "cython<3" "numpy<2"
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
export CFLAGS="-isysroot $SDKROOT -arch arm64"
export LDFLAGS="-isysroot $SDKROOT -arch arm64"
python -m pip install --no-cache-dir xtcocotools==1.14.3

mim install mmengine
mim install "mmcv==2.1.0"
mim install "mmpose==1.3.2"
mim install "mmdet==3.2.0"