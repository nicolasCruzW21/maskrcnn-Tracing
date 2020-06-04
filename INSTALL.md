## Installation

### Requirements:
- PyTorch 1.5
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.0


### Option 1: Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create -n maskrcnn-Trace python=3
conda activate maskrcnn-Trace

# this installs the right pip and dependencies for the fresh python
conda install ipython pip

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.1
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install cityscapesScripts
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install OPENCV from source

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/nicolasCruzW21/maskrcnn-Tracing.git
cd maskrcnn-Tracing/maskrcnn_benchmark/
mkdir lib
cd ..

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop
cd maskrcnn_benchmark/lib/

unset INSTALL_DIR

#To trace a model:
cd demo/
python trace.py
