import argparse
from datasets import load_from_disk
from data.preprocess.stats import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Dataset statistics")
    parser.add_argument("--dataset", type=str, required=True, help="Directory of the dataset (output of abs2ext.py)")
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset)

    tot_documents = 0
    tot_doc_sentences = 0
    tot_doc_tokens = 0
    tot_ref_summ_tokens = 0
    tot_ext_summ_tokens = 0
    tot_ext_summ_sentences = 0

    for split in dataset:
        print(f">>>  {split}  {'>'*60}")
        documents = len(dataset[split])
        doc_sentences = documentsSentences(dataset[split])
        doc_tokens = documentsTokens(dataset[split])
        ref_summ_tokens = referenceSummariesTokens(dataset[split])
        ext_summ_tokens = extractiveSummariesTokens(dataset[split])
        ext_summ_sentences = extractiveSummariesSentences(dataset[split])

        tot_documents += documents
        tot_doc_sentences += doc_sentences
        tot_doc_tokens += doc_tokens
        tot_ref_summ_tokens += ref_summ_tokens
        tot_ext_summ_tokens += ext_summ_tokens
        tot_ext_summ_sentences += ext_summ_sentences

        print(f"{'Number of documents'.rjust(40)} {documents}")
        print(f"{'Average document sentences'.rjust(40)} {(doc_sentences / documents):.2f}")
        print(f"{'Average document tokens'.rjust(40)} {(doc_tokens / documents):.2f}")
        print(f"{'Average reference summary tokens'.rjust(40)} {(ref_summ_tokens / documents):.2f}")
        print(f"{'Average extractive summary tokens'.rjust(40)} {(ext_summ_tokens / documents):.2f}")
        print(f"{'Average extractive summary sentences'.rjust(40)} {(ext_summ_sentences / documents):.2f}")

    print(f">>>  Overall  {'>'*60}")
    print(f"{'Number of documents'.rjust(40)} {tot_documents}")
    print(f"{'Average document sentences'.rjust(40)} {(tot_doc_sentences / tot_documents):.2f}")
    print(f"{'Average document tokens'.rjust(40)} {(tot_doc_tokens / tot_documents):.2f}")
    print(f"{'Average reference summary tokens'.rjust(40)} {(tot_ref_summ_tokens / tot_documents):.2f}")
    print(f"{'Average extractive summary tokens'.rjust(40)} {(tot_ext_summ_tokens / tot_documents):.2f}")
    print(f"{'Average extractive summary sentences'.rjust(40)} {(tot_ext_summ_sentences / tot_documents):.2f}")
