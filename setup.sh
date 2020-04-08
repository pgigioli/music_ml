#!/bin/bash
#conda install -c conda-forge librosa -y

pip install librosa --user
cd ../
wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz
tar -xzf libsndfile-1.0.28.tar.gz
cd libsndfile-1.0.28
./configure --prefix=/usr --disable-static --docdir=/usr/share/doc/libsndfile-1.0.28
sudo make install
cd ../
rm libsndfile-1.0.28.tar.gz
sudo yum install alsa-lib-devel -y
pip install magenta --user