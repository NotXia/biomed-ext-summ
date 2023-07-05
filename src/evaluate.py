from models.loader import loadModel
from datasets import load_from_disk
import torch
import numpy as np
import argparse
import os
import random
from metrics.rouge import evalROUGE
from metrics.logger import MetricsLogger
import sys


"""
    Model evaluation.

    Parameters
    ----------
        model : Model|str

        datasets : DatasetDict

        splits : str[]
            Splits of the dataset to use

        strategy : str
            See BaseSummarizer.

        strategy_args : any
            See BaseSummarizer.
"""
def evaluate(model, datasets, splits, strategy, strategy_args):
    metrics = MetricsLogger()

    if type(model) != str: model.eval()
    with torch.no_grad():
        for split in splits:
            for i, document in enumerate(datasets[split]):
                summary = ""

                if model == "oracle":
                    selected_sents = [sent for i, sent in enumerate(document["sentences"]) if document["labels"][i]]
                    summary = "\n".join(selected_sents)
                else:
                    selected_sents, _ = model.summarizeSentences(document["sentences"], strategy, strategy_args)
                    summary = "\n".join(selected_sents)

                metrics.add("rouge", evalROUGE( [document["ref_summary"]], [summary] ))
                sys.stdout.write(f"\r{i+1}/{len(datasets[split])} ({split}) --- {metrics.format(['rouge'])}")
                sys.stdout.flush()

    sys.stdout.write(f"\r")
    print(f"{metrics.format(['rouge'])}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model evaluation")
    model_args = parser.add_mutually_exclusive_group(required=True)
    model_args.add_argument("--checkpoint", type=str, help="Path to the checkpoint of the model")
    model_args.add_argument("--oracle", action="store_true", help="Evaluate using the oracle (greedy selection)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a preprocessed extractive dataset")
    parser.add_argument("--splits", type=str, default="test", required=True, help="Splits of the dataset to use for the evaluation (e.g. test,validation)")
    strategy_args = parser.add_mutually_exclusive_group(required="--checkpoint" in sys.argv)
    strategy_args.add_argument("--strategy-length", type=int, help="Summary generated with a length upper bound")
    strategy_args.add_argument("--strategy-count", type=int, help="Summary generated by selecting a given number of sentences")
    strategy_args.add_argument("--strategy-ratio", type=float, help="Summary proportional to the size of the document")
    strategy_args.add_argument("--strategy-threshold", type=float, help="Summary generated by selecting sentences with a score greater or equal to the given value [0, 1]")
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(mode=True)
    if torch.cuda.is_available(): os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    print("-- Loading model --")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = load_from_disk(args.dataset)
    model = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = loadModel(checkpoint["model_name"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif args.oracle:
        model = "oracle"
    
    strategy = None
    strategy_args = None
    if args.strategy_length is not None:
        strategy = "length"
        strategy_args = args.strategy_length
    elif args.strategy_count is not None:
        strategy = "count"
        strategy_args = args.strategy_count
    elif args.strategy_ratio is not None:
        strategy = "ratio"
        strategy_args = args.strategy_ratio
    elif args.strategy_threshold is not None:
        strategy = "threshold"
        strategy_args = args.strategy_threshold
    else:
        raise NotImplementedError
    

    print("-- Starting evaluation --")
    evaluate(
        model = model,
        datasets = datasets,
        splits = args.splits.split(","),
        strategy = strategy,
        strategy_args = strategy_args
    )
