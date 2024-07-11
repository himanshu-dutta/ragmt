#! /bin/bash

python /home/src/rag_exp.py \
 --documents_dataset_path /home/data/koran/datastore.en/dataset \
 --documents_index_path /home/data/koran/datastore.en/hnsw_index.faiss \
 --query_encoder_name_or_path facebook/dpr-ctx_encoder-multiset-base \
 --generator_model_name_or_path facebook/bart-large \
 --data-root /home/data/koran \
 --output-root /home/runs/ragmt\
 --domain koran \
 --src-lng en \
 --tgt-lng de \
 --batch-size 1 \
 --epochs 10