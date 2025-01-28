#!/bin/bash

# Updated for Laphet LM Toolkit Version 0.7
# Last updated: 28 Jan 2025

# Create the output and log directories if they don't exist
mkdir -p model/tag/
mkdir -p output/tag/
mkdir -p log/tag/

# Function to train, generate text, and test a language model
task() {
  local model_type=$1
  local model_file="./model/tag/${model_type}.nofz.model"
  local output_file="./output/tag/${model_type}_nofz_gen_texts.txt"
  local log_file="./log/tag/${model_type}.nofz.log"
  local train_data="./data/myPOS/tag/train_tag.txt"
  local dev_data="./data/myPOS/tag/dev_tag.txt"
  local test_data="./data/myPOS/tag/test_tag.txt"
  local start_name="./data/myPOS/tag/start_tags.txt"

  {
    echo "Training ${model_type^} language model:";
    time python -u laphet.py --model_type $model_type --train --data $train_data \
      --dev_file $dev_data --model $model_file --seq_len 50 --epochs 10 --batch_size 32 \
      --lr 0.0001 --embedding_method fasttext_no_freeze \
      --fasttext_model ./fasttext-model/mypos.tag.100.bin --embed_dim 100;

    echo "Text generation:";
    time python -u laphet.py --model_type $model_type --generate --model $model_file \
      --seq_len 50 --prompt "n" --no_of_generation 10 \
      --embedding_method fasttext_no_freeze

    echo "Batch text generation from file:";
    time python -u laphet.py --model_type $model_type --generate --model $model_file \
      --seq_len 2 --input $start_name --no_of_generation 5 --output $output_file \
      --embedding_method fasttext_no_freeze;

    echo "Testing:";
    time python -u laphet.py --model_type $model_type --test --model $model_file \
      --test_file $test_data --seq_len 50 --batch_size 64 --embedding_method fasttext_no_freeze 2>&1;
  } | tee "$log_file"
}

# Run tasks for each model type in the specified order
#task mlp
#task bilstm
#task transformer
#task bert
task gpt

echo "All tasks completed!"

