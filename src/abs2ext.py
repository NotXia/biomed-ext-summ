"""
Abstractive to extractive conversion.
"""

import argparse
from data.preprocess.abs2ext import parseAbs2Ext
from rouge_score.rouge_scorer import RougeScorer



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Dataset preprocessing - Abstractive to extractive dataset conversion")
    parser.add_argument("--dataset", type=str, choices=["cnn_dailymail", "ms2", "cochrane", "sumpubmed"], required=True)
    parser.add_argument("--dataset-dir", type=str, help="Directory of the dataset, if not specified, it will be downloaded")
    parser.add_argument("--head", action="store_true", help="Show some rows of the parsed dataset")
    parser.add_argument("--output", type=str, help="Directory where the dataset will be exported to")
    parser.add_argument("--proc", type=int, default=1, help="Number of processes to create")
    parser.add_argument("--selection-size", type=int, default=3, help="Number of sentences to select. The maximum between this value and the number of sentences in the abstractive summary will be considered.")
    args = parser.parse_args()
    

    parsed_dataset = parseAbs2Ext(args.dataset, args.dataset_dir, args.selection_size, args.proc)


    # Save dataset locally
    if args.output != None:
        parsed_dataset.save_to_disk(args.output)


    # Print some of the parsed data
    if args.head:
        scorer = RougeScorer(["rouge1", "rouge2", "rougeL"])
        dataset = parsed_dataset[ list(parsed_dataset.keys())[0] ]
        
        for i in range(min(5, len(dataset))):
            sentences = dataset[i]["sentences"]
            extractive_summ = "\n".join([ s for j, s in enumerate(sentences) if dataset[i]["labels"][j] ])
            rouge = scorer.score(dataset[i]["ref_summary"], extractive_summ)
            
            print("--- Document sentences ---")
            print("\n".join(sentences))
            print("--------------------------\n")
            print("--- Abstractive summary ---")
            print(dataset[i]["ref_summary"])
            print("---------------------------\n")
            print("--- Extractive summary ---")
            print(extractive_summ)
            print("--------------------------\n")
            print("--- ROUGE ---")
            print(f"ROUGE-1 \t P={rouge['rouge1'].precision:.4f}\t R={rouge['rouge1'].recall:.4f} \t F1={rouge['rouge1'].fmeasure:.4f}")
            print(f"ROUGE-2 \t P={rouge['rouge2'].precision:.4f}\t R={rouge['rouge2'].recall:.4f} \t F1={rouge['rouge2'].fmeasure:.4f}")
            print(f"ROUGE-L \t P={rouge['rougeL'].precision:.4f}\t R={rouge['rougeL'].recall:.4f} \t F1={rouge['rougeL'].fmeasure:.4f}")
            print("-------------")
            print()
