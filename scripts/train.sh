#! /bin/bash

# EXPERIMENT NAME FORMAT: <DATASTORE_TYPE>-<TASK>-<DATASET>-<RETRIEVER_ENCODER_NAME>-<GENERATOR_NAME>-<SRC_LANG>-<TGT_LANG>

CUDA_VISIBLE_DEVICES=1,4,7 PYTHONPATH=. python ragmt/training.py \
  --experiment_name kg-domainadaptation-koran-dpr_ctx_encoder-nllb200_600M-En-De \
  --query_encoder_model_path facebook/dpr-ctx_encoder-multiset-base \
  --generator_model_path /home/artifacts/model/pretrained/nllb200-600M \
  --kb_dataset_dir /home/artifacts/scratch/koran_kb/dataset \
  --kb_index_path /home/artifacts/scratch/koran_kb/index.hnswflat.faiss \
  --dataset_dir /home/artifacts/data/koran \
  --source_lang english \
  --target_lang german \
  --train_batch_size 2 \
  --val_batch_size 2 \
  --use_retrieval \
  --num_gpus 3 \
  --accumulate_grad_batches 4 \
  --use_fp16