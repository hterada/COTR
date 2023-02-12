#!/bin/bash

docker run --rm --gpus all -v /home/terada/work/COTR:/COTR \
  -v $HOME/.Xauthority:/root/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  --net host \
  --expose 8888 \
  --expose 80 \
  -p 192.168.128.27:80:8888 \
  -it 'cotr/demo:1.0-cuda12.0.0' bash
