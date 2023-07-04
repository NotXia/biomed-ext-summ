import argparse
import random
from data.preprocess.bert import preprocessForBERT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Dataset preprocessing - Parse for BERT")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory of the dataset (output of abs2ext.py)")
    parser.add_argument("--output", type=str, required=True, help="Directory where the dataset will be exported to")
    parser.add_argument("--proc", type=int, default=1, help="Number of processes to create")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model the dataset targets (e.g. bert-base-uncased)")
    parser.add_argument("--model-family", type=str, choices=["bert"], default="bert", help="Family of the model (e.g. bert)")
    args = parser.parse_args()
    
    random.seed(42)


    parsed_dataset = None

    if args.model_family == "bert":
        parsed_dataset = preprocessForBERT(args.model, args.dataset_dir, args.proc)
    else:
        raise NotImplementedError


    # Save dataset locally
    if args.output != None:
        parsed_dataset.save_to_disk(args.output)