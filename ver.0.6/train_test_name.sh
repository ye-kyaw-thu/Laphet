#!/bin/bash

# Create the output and log directories if they don't exist
mkdir -p model/name/
mkdir -p output/name/
mkdir -p log/name/

# Function to train, generate text, and test a language model
task() {
  local model_type=$1
  local model_file="./model/name/${model_type}.model"
  local output_file="./output/name/${model_type}_gen_texts.txt"
  local log_file="./log/name/${model_type}.log"
  local train_data="./data/myRoman/train_name.txt"
  local dev_data="./data/myRoman/dev_name.txt"
  local test_data="./data/myRoman/test_name.txt"
  local start_name="./data/myRoman/start_names.txt"

  {
    echo "Training ${model_type^} language model:";
    time python -u laphet.py --model_type $model_type --train --data $train_data \
      --dev_file $dev_data --model $model_file --seq_len 50 --epochs 10 --batch_size 32 --lr 0.0001;

    echo "Text generation:";
    time python -u laphet.py --model_type $model_type --generate --model $model_file \
      --seq_len 50 --prompt "ရဲ" --no_of_generation 10;

    echo "Batch text generation from file:";
    time python -u laphet.py --model_type $model_type --generate --model $model_file \
      --seq_len 2 --input $start_name --no_of_generation 5 --output $output_file;

    echo "Testing:";
    time python -u laphet.py --model_type $model_type --test --model $model_file \
      --test_file $test_data --seq_len 50 --batch_size 64 2>&1;
  } | tee "$log_file"
}

# Run tasks for each model type in the specified order
task mlp
task bilstm
task transformer
task bert
task gpt

echo "All tasks completed!"

