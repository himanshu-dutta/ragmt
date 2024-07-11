#! /bin/bash

PYTHONPATH=. python ragmt/data/build_kb.py \
  --tsv_path /home/data/koran/kg.en.csv \
  --output_dir /home/artifacts/scratch/koran_kb \
  --document_encoder_path_or_name facebook/dpr-ctx_encoder-multiset-base \
  --batch_size 32 \
  --d 768 \
  --m 128