#! /bin/bash

docker run -it \
 --gpus all \
 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
 --rm \
 --mount type=bind,source="$(pwd)"/,target=/home/ \
 -p 9090:9090 \
 -d \
 --name ragmt-dev \
 himanshu/ragmt-dev