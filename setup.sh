#!/bin/bash

# install resource monitor
sudo yum install htop -y

# install librosa
pip install librosa --user

# manually install sndfile
cd ../
wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz
tar -xzf libsndfile-1.0.28.tar.gz
cd libsndfile-1.0.28
./configure --prefix=/usr --disable-static --docdir=/usr/share/doc/libsndfile-1.0.28
sudo make install
cd ../
rm libsndfile-1.0.28.tar.gz

# install magenta
sudo yum install alsa-lib-devel -y
pip install magenta --user

# download wavenet weights
cd music_ml
wget http://download.magenta.tensorflow.org/models/nsynth/wavenet-ckpt.tar
tar -xvf wavenet-ckpt.tar
rm wavenet-ckpt.tar