sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy
cuda-install-samples-7.5.sh ~/
cd NVIDIA_CUDA-7.5_Samples/1_Utilities/deviceQuery
make
./deviceQuery
echo -e "\n[global]\nfloatX=float32\ndevice=gpu\n[mode]=FAST_RUN\n\n[nvcc]\nfastmath=True\n\n[cuda]\nroot=/usr/local/cuda" >> ~/.theanorc
cd ~
nano theano_gpu_test.py     # copied example theano gpu test from tutorial
python theano_gpu_test.py
# Copied cudnn file from Maciej's computer
tar xvfz cudnn-7.0-linux-x64-v4.0-rc.tgz
echo $CUDA_ROOT

cd /user/local/cuda
sudo cp ~/cuda/include/cudnn.h ./include/
sudo cp ~/cuda/lib64/libcudnn.so ./lib64/
sudo cp ~/cuda/lib64/libcudnn.so.4 ./lib64/
sudo cp ~/cuda/lib64/libcudnn.so.4.0.4 ./lib64/
sudo cp ~/cuda/lib64/libcudnn_static.a ./lib64/

cd ~
git clone https://github.com/Lasagne/Lasagne.git
cd ~/Lasagne/
sudo pip install .
python -c "from theano.sandbox.cuda.dnn import dnn_available as d; print(d() or d.msg)"
cd ~
wget http://www.arcadelearningenvironment.org/wp-content/uploads/2015/10/Arcade-Learning-Environment-0.5.1.zip
sudo apt-get install unzip
unzip Arcade-Learning-Environment-0.5.1.zip
sudo apt-get install cmake libsdl1.2-dev
cd Arcade-Learning-Environment-0.5.1/
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF .
make -j 4
sudo pip install .

python mnist.py cnn



cd ~
git clone https://github.com/NVIDIA/cnmem.git cnmem
cd cnmem/
mkdir build
cd build
cmake ..
make
cd /usr/local/cuda
sudo cp ~/cnmem/include/cnmem.h  ./include
sudo mv ~/cnmem/build/libcnmem.so ./lib64
