from models.loader import loadModel
import torch
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint of the model")
    document_args = parser.add_mutually_exclusive_group(required=True)
    document_args.add_argument("--document", type=str, help="Document to summarize")
    document_args.add_argument("--document-path", type=str, help="Path to the document to summarize")
    strategy_args = parser.add_mutually_exclusive_group(required=True)
    strategy_args.add_argument("--strategy-length", type=int, help="Summary generated with a length upper bound")
    strategy_args.add_argument("--strategy-count", type=int, help="Summary generated by selecting a given number of sentences")
    strategy_args.add_argument("--strategy-ratio", type=float, help="Summary proportional to the size of the document")
    strategy_args.add_argument("--strategy-threshold", type=float, help="Summary generated by selecting sentences with a score greater or equal to the given value [0, 1]")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = loadModel(checkpoint["model_name"], checkpoint["model_family"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    document = args.document
    if args.document_path != None:
        with open(args.document_path, "r") as f:
            document = f.read()

    summary_sents = []

    if args.strategy_length != None:
        summary_sents, _ = model.summarize(document, "length", args.strategy_length)
    elif args.strategy_count != None:
        summary_sents, _ = model.summarize(document, "count", args.strategy_count)
    elif args.strategy_ratio != None:
        summary_sents, _ = model.summarize(document, "ratio", args.strategy_ratio)
    elif args.strategy_threshold != None:
        summary_sents, _ = model.summarize(document, "threshold", args.strategy_threshold)

    summary = "\n".join(summary_sents)
    print()
    print(summary)
    print()
    print(document)
