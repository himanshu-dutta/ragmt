#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. streamlit run --server.address '0.0.0.0' --server.port 9090 --server.fileWatcherType none /home/demo/main.py