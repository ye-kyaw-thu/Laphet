"""
Laphet language model toolkit (version 0.7)
Written by Ye Kyaw Thu, LU Lab., Myanmar.
Last Updated: 29 Jan 2025.

How to run:

python laphet.py --help
bash ./train_test_name.sh
bash ./train_test_name_ftfz.sh
bash ./train_test_name_nofz.sh 

bash ./train_test_tag.sh
bash ./train_test_tag_ftfz.sh
bash ./train_test_tag_nofz.sh
"""

import argparse
import os
import torch
import random

from lm_lib.util import Tokenizer, BurmeseDataset #, calculate_perplexity
from lm_lib.mlp_lm import MLP, train_mlp, test_mlp
from lm_lib.bilstm_lm import BiLSTM, train_bilstm, test_bilstm
from lm_lib.transformer_lm import TRANSFORMER, train_transformer, test_transformer
from lm_lib.bert_lm import BERT, train_bert, test_bert
from lm_lib.gpt_lm import GPT, train_gpt, test_gpt
from gensim.models.fasttext import load_facebook_model

# Main Function

def main():
    parser = argparse.ArgumentParser(description="Laphet language model toolkit.")
    parser.add_argument("--model_type", type=str, required=True, choices=["mlp", "bilstm", "transformer","bert", "gpt"],
                        help="Type of model to use: mlp, bilstm, transformer, bert or gpt.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--generate", action="store_true", help="Generate text using the trained model.")
    parser.add_argument("--test", action="store_true", help="Test the BERT model and evaluate perplexity.")
    parser.add_argument("--data", type=str, help="Path to the dataset.")
    parser.add_argument("--model", type=str, help="Path to save/load the model.")
    parser.add_argument("--vocab", type=str, help="Path to save/load the tokenizer vocabulary")
    parser.add_argument("--dev_file", type=str, help="Path to the development or validation dataset.")
    parser.add_argument("--test_file", type=str, help="Path to the test dataset.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for text generation (default: None).")
    parser.add_argument("--input", type=str, default=None, help="File with starting words for line-by-line generation (default: None).")
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length (default: 30).")
    parser.add_argument("--output", type=str, default=None, help="File to save the generated text (default: None).")
    parser.add_argument("--no_of_generation", type=int, default=1, help="Number of sequences to generate (default: 1).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001).")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension (default: 256).")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads (default: 4).")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers (default: 4).")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for LSTM (default: 512).")
    parser.add_argument("--ff_dim", type=int, default=512, help="Feedforward dimension for Transformer (default: 512).")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for LSTM (default: 0.5).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0).")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling (default: 10).")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling (default: 0.9).")
    parser.add_argument("--embedding_method", type=str, default="nn.Embedding", choices=["nn.Embedding", "fasttext_freeze", "fasttext_no_freeze"],
                    help="Embedding method to use: nn.Embedding, fasttext_freeze, or fasttext_no_freeze.")
    parser.add_argument("--fasttext_model", type=str, default=None,
                    help="Path to the pretrained FastText model (required if embedding_method is fasttext_freeze or fasttext_no_freeze).")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        # Load training and validation datasets
        with open(args.data, "r", encoding="utf-8") as f:
            train_texts = f.read().splitlines()

        # Load validation dataset
        with open(args.dev_file, "r", encoding="utf-8") as f:
            dev_texts = f.read().splitlines()

        tokenizer = Tokenizer()
        tokenizer.fit(train_texts)

        train_dataset = BurmeseDataset(train_texts, tokenizer, args.seq_len)
        dev_dataset = BurmeseDataset(dev_texts, tokenizer, args.seq_len)

        # Ensure the directory for the model exists
        model_dir = os.path.dirname(args.model)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save tokenizer information
        vocab_file_path = args.vocab if args.vocab else args.model + ".vocab"
        with open(vocab_file_path, "w", encoding="utf-8") as vocab_file:
            for token, idx in tokenizer.token_to_id.items():
                vocab_file.write(f"{token}\t{idx}\n")

        if args.model_type == "mlp":
            #model = MLP(len(tokenizer.token_to_id), args.embed_dim, args.hidden_dim, args.seq_len)
            model = MLP(
                vocab_size=len(tokenizer.token_to_id),
                embedding_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                seq_len=args.seq_len,
                embedding_method=args.embedding_method,
                fasttext_model_path=args.fasttext_model,
                tokenizer=tokenizer
            )
            train_mlp(model, train_dataset, dev_dataset, args.epochs, args.batch_size, args.lr, device, args.model, embedding_method=args.embedding_method, fasttext_model_path=args.fasttext_model)

        elif args.model_type == "bert":
            model = BERT(
                vocab_size=len(tokenizer.token_to_id),
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                ff_dim=args.ff_dim,
                seq_len=args.seq_len,
                embedding_method=args.embedding_method,
                fasttext_model_path=args.fasttext_model,
                tokenizer=tokenizer
            )
            train_bert(model, train_dataset, dev_dataset, args.epochs, args.batch_size, args.lr, device, args.model, embedding_method=args.embedding_method, fasttext_model_path=args.fasttext_model)

        elif args.model_type == "bilstm":

            model = BiLSTM(
                vocab_size=len(tokenizer.token_to_id),
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                embedding_method=args.embedding_method,
                fasttext_model_path=args.fasttext_model,
                tokenizer=tokenizer
            )
            train_bilstm(model, train_dataset, dev_dataset, args.epochs, args.batch_size, args.lr, device, args.model, embedding_method=args.embedding_method, fasttext_model_path=args.fasttext_model)

        elif args.model_type == "gpt":
            model = GPT(
                vocab_size=len(tokenizer.token_to_id),
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                ff_dim=args.ff_dim,
                seq_len=args.seq_len,
                embedding_method=args.embedding_method,
                fasttext_model_path=args.fasttext_model,
                tokenizer=tokenizer
            )
            train_gpt(model, train_dataset, dev_dataset, args.epochs, args.batch_size, args.lr, device, args.model, embedding_method=args.embedding_method, fasttext_model_path=args.fasttext_model)

        elif args.model_type == "transformer":
            model = TRANSFORMER(
                vocab_size=len(tokenizer.token_to_id),
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                ff_dim=args.ff_dim,
                seq_len=args.seq_len,
                embedding_method=args.embedding_method,
                fasttext_model_path=args.fasttext_model,
                tokenizer=tokenizer
            )
            train_transformer(model, train_dataset, dev_dataset, args.epochs, args.batch_size, args.lr, device, args.model, embedding_method=args.embedding_method, fasttext_model_path=args.fasttext_model)

    if args.generate:
        if not args.model:
            raise ValueError("Model path must be specified for generation mode.")

        # Load the model
        checkpoint = torch.load(args.model)
        tokenizer = Tokenizer()
        tokenizer.token_to_id = checkpoint["token_to_id"]
        tokenizer.id_to_token = checkpoint["id_to_token"]

        if args.model_type == "mlp":
            model = MLP(
                vocab_size=len(tokenizer.token_to_id),
                embedding_dim=checkpoint["embed_dim"],
                hidden_dim=checkpoint["hidden_dim"],
                seq_len=checkpoint["seq_len"],
                embedding_method=checkpoint["embedding_method"],  # Load embedding method
                fasttext_model_path=checkpoint["fasttext_model_path"],  # Load FastText path
                tokenizer=tokenizer  # Pass the tokenizer object
            )

        elif args.model_type == "bilstm":
            model = BiLSTM(
                vocab_size=len(tokenizer.token_to_id),
                embed_dim=checkpoint["embed_dim"],
                hidden_dim=checkpoint["hidden_dim"],
                num_layers=checkpoint["num_layers"],
                dropout=checkpoint["dropout"],
                embedding_method=checkpoint["embedding_method"],  # Load embedding method
                fasttext_model_path=checkpoint["fasttext_model_path"],  # Load FastText path
                tokenizer=tokenizer  # Pass the tokenizer object
            )

        elif args.model_type == "transformer":
            model = TRANSFORMER(
                vocab_size=len(tokenizer.token_to_id),
                embed_dim=checkpoint["embed_dim"],
                num_heads=checkpoint["num_heads"],
                num_layers=checkpoint["num_layers"],
                ff_dim=checkpoint["ff_dim"],
                seq_len=checkpoint["seq_len"],
                embedding_method=checkpoint["embedding_method"],  # Load embedding method
                fasttext_model_path=checkpoint["fasttext_model_path"],  # Load FastText path
                tokenizer=tokenizer  # Pass the tokenizer object
            )

        elif args.model_type == "bert":
            model = BERT(
                vocab_size=len(tokenizer.token_to_id),
                embed_dim=checkpoint["embed_dim"],
                num_heads=checkpoint["num_heads"],
                num_layers=checkpoint["num_layers"],
                ff_dim=checkpoint["ff_dim"],
                seq_len=checkpoint["seq_len"],
                embedding_method=checkpoint["embedding_method"],  # Load embedding method
                fasttext_model_path=checkpoint["fasttext_model_path"],  # Load FastText path
                tokenizer=tokenizer  # Pass the tokenizer object
            )

        elif args.model_type == "gpt":
            model = GPT(
                vocab_size=len(tokenizer.token_to_id),
                embed_dim=checkpoint["embed_dim"],
                num_heads=checkpoint["num_heads"],
                num_layers=checkpoint["num_layers"],
                ff_dim=checkpoint["ff_dim"],
                seq_len=checkpoint["seq_len"],
                embedding_method=checkpoint["embedding_method"],  # Load embedding method
                fasttext_model_path=checkpoint["fasttext_model_path"],  # Load FastText path
                tokenizer=tokenizer  # Pass the tokenizer object
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        generated_texts = []

        if args.input:
            # Read starting words from the input file
            with open(args.input, "r", encoding="utf-8") as f:
                starting_words = [line.strip() for line in f.readlines()]
        else:
            # Use prompt or random starting word
            starting_words = [args.prompt] if args.prompt else [None]

        for start_word in starting_words:
            for _ in range(args.no_of_generation):
                if not start_word:
                    # Generate a random initial prompt
                    random_id = random.randint(0, len(tokenizer.id_to_token) - 1)
                    start_word = tokenizer.id_to_token[random_id]
                    print(f"Random Prompt Generated: {start_word	}")

                # Encode the starting word
                input_ids = tokenizer.encode(start_word, return_tensors="pt").to(device)

                # Generate text
                generated_ids = model.generate(
                    input_ids,
                    max_length=args.seq_len,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                generated_text = tokenizer.decode(generated_ids.squeeze().tolist())
                generated_texts.append(generated_text)

        # Save or print all generated texts
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for text in generated_texts:
                    f.write(text + "\n")
            print(f"Generated texts saved to {args.output}")
        else:
            for idx, text in enumerate(generated_texts, start=1):
                print(f"Generated Text {idx}: {text}")

    if args.test:
        if not args.model:
            raise ValueError("Model path must be specified for testing mode.")
        if not args.test_file:
            raise ValueError("Test file must be specified for testing mode.")

        # Load the model
        checkpoint = torch.load(args.model)
        tokenizer = Tokenizer()
        tokenizer.token_to_id = checkpoint["token_to_id"]
        tokenizer.id_to_token = checkpoint["id_to_token"]

        if args.model_type == "mlp":
            model = MLP(
                vocab_size=len(tokenizer.token_to_id),
                embedding_dim=checkpoint["embed_dim"],
                hidden_dim=checkpoint["hidden_dim"],
                seq_len=checkpoint["seq_len"],
                embedding_method=checkpoint["embedding_method"],  # Load embedding method
                fasttext_model_path=checkpoint["fasttext_model_path"],  # Load FastText path
                tokenizer=tokenizer  # Pass the tokenizer object
            )

        elif args.model_type == "bilstm":
            model = BiLSTM(
                vocab_size=len(tokenizer.token_to_id),
                embed_dim=checkpoint["embed_dim"],
                hidden_dim=checkpoint["hidden_dim"],
                num_layers=checkpoint["num_layers"],
                dropout=checkpoint["dropout"],
                embedding_method=checkpoint["embedding_method"],  # Load embedding method
                fasttext_model_path=checkpoint["fasttext_model_path"],  # Load FastText path
                tokenizer=tokenizer  # Pass the tokenizer object
            )

        elif args.model_type == "transformer":
            model = TRANSFORMER(
                vocab_size=len(tokenizer.token_to_id),
                embed_dim=checkpoint["embed_dim"],
                num_heads=checkpoint["num_heads"],
                num_layers=checkpoint["num_layers"],
                ff_dim=checkpoint["ff_dim"],
                seq_len=checkpoint["seq_len"],
                embedding_method=checkpoint["embedding_method"],  # Load embedding method
                fasttext_model_path=checkpoint["fasttext_model_path"],  # Load FastText path
                tokenizer=tokenizer  # Pass the tokenizer object
            )

        elif args.model_type == "bert":
            model = BERT(
                vocab_size=len(tokenizer.token_to_id),
                embed_dim=checkpoint["embed_dim"],
                num_heads=checkpoint["num_heads"],
                num_layers=checkpoint["num_layers"],
                ff_dim=checkpoint["ff_dim"],
                seq_len=checkpoint["seq_len"],
                embedding_method=checkpoint["embedding_method"],  # Load embedding method
                fasttext_model_path=checkpoint["fasttext_model_path"],  # Load FastText path
                tokenizer=tokenizer  # Pass the tokenizer object
            )

        elif args.model_type == "gpt":
            model = GPT(
                vocab_size=len(tokenizer.token_to_id),
                embed_dim=checkpoint["embed_dim"],
                num_heads=checkpoint["num_heads"],
                num_layers=checkpoint["num_layers"],
                ff_dim=checkpoint["ff_dim"],
                seq_len=checkpoint["seq_len"],
                embedding_method=checkpoint["embedding_method"],  # Load embedding method
                fasttext_model_path=checkpoint["fasttext_model_path"],  # Load FastText path
                tokenizer=tokenizer  # Pass the tokenizer object
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        # Run testing and evaluation
        if args.model_type == "mlp":
            test_mlp(model, tokenizer, args.test_file, args.seq_len, args.batch_size, device)
        elif args.model_type == "bilstm":
            test_bilstm(model, tokenizer, args.test_file, args.seq_len, args.batch_size, device)
        elif args.model_type == "transformer":
            test_transformer(model, tokenizer, args.test_file, args.seq_len, args.batch_size, device)
        elif args.model_type == "bert":
            test_bert(model, tokenizer, args.test_file, args.seq_len, args.batch_size, device)
        elif args.model_type == "gpt":
            test_gpt(model, tokenizer, args.test_file, args.seq_len, args.batch_size, device)

if __name__ == "__main__":
    main()
    
