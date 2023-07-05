import argparse
import random
from transformers import AutoTokenizer, BertTokenizer, BertTokenizerFast
from data.preprocess.bert import preprocessForBERT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Dataset preprocessing - Parse for BERT")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory of the dataset (output of abs2ext.py)")
    parser.add_argument("--output", type=str, required=True, help="Directory where the dataset will be exported to")
    parser.add_argument("--proc", type=int, default=1, help="Number of processes to create")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model the dataset targets (e.g. bert-base-uncased)")
    args = parser.parse_args()
    
    random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    parsed_dataset = None

    if isinstance(tokenizer, BertTokenizer) or isinstance(tokenizer, BertTokenizerFast):
        parsed_dataset = preprocessForBERT(tokenizer, args.dataset_dir, args.proc)
    else:
        raise NotImplementedError


    # Save dataset locally
    if args.output != None:
        parsed_dataset.save_to_disk(args.output)