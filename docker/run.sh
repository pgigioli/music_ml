nvidia-docker run -it --rm -p 8888:8888 -v $(dirname `pwd`):/$(basename $(dirname `pwd`)) music_ml:pytorch-gpu
