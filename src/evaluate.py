from models.bert_summ import BERTSummarizer
from data.dataset_loader import load_dataset
from transformers import BertTokenizer
import torch
from tqdm import tqdm
import numpy as np
import argparse
import os
import random
from metrics.accuracy import accuracy
from metrics.rouge import evalROUGE
from metrics.logger import MetricsLogger



"""
    Model evaluation.

    Parameters
    ----------
        model : Model
        dataloader : DataLoader
        device : Device
        tokenizer : BertTokenizer
"""
def evaluate(model, dataloader, device, tokenizer):
    metrics = MetricsLogger()

    model.eval()
    with torch.no_grad():
        for documents, labels in tqdm(dataloader):
            labels = labels.to(device)

            outputs = model.predict(documents, device)

            metrics.add("accuracy", accuracy(labels, outputs))
            metrics.add("rouge", evalROUGE(model, tokenizer, documents, labels, outputs))

    print(f"Val: {metrics.format(['accuracy', 'rouge'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint of the model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a preprocesses dataset")
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(mode=True)
    if torch.cuda.is_available(): os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    print("-- Loading model --")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = BERTSummarizer(checkpoint["model_name"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = BertTokenizer.from_pretrained(checkpoint["model_name"])
    datasets = load_dataset(args.dataset, tokenizer=tokenizer, splits=["test"])
    dataloader = torch.utils.data.DataLoader(datasets["test"], batch_size=args.batch_size)

    print("-- Starting evaluation --")
    evaluate(
        model = model,
        dataloader = dataloader,
        tokenizer = tokenizer,
        device = device
    )
