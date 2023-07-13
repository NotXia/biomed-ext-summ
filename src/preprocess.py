"""
Parse a dataset for a specific model
Based on the paper: Text Summarization with Pretrained Encoders.
"""
import argparse
import random
from datasets import DatasetDict, load_from_disk
from data.preprocess.loader import loadPreprocessUtilities



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Dataset - model-specific preprocessing")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory of the dataset (output of abs2ext.py)")
    parser.add_argument("--output", type=str, required=True, help="Directory where the dataset will be exported to")
    parser.add_argument("--proc", type=int, default=1, help="Number of processes to create")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model the dataset targets (e.g. bert-base-uncased)")
    args = parser.parse_args()
    
    random.seed(42)

    dataset = load_from_disk(args.dataset_dir)
    datasetMapFn, datasetFilterFn = None, None
    parsed_dataset = None
    datasetMapFn, datasetFilterFn = loadPreprocessUtilities(args.model)

    dataset = dataset.map(datasetMapFn, num_proc=args.proc)
    parsed_dataset = {
        "train": datasetFilterFn(dataset["train"]),
        "test": datasetFilterFn(dataset["test"]),
        "validation": datasetFilterFn(dataset["validation"])
    }
    parsed_dataset = DatasetDict(parsed_dataset)
    
    if args.output != None:
        parsed_dataset.save_to_disk(args.output)