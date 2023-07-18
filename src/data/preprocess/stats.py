"""
Dataset statistics
"""
import re
from tqdm import tqdm



def documentsSentences(dataset):
    tot_sentences = 0
    for data in tqdm(dataset, desc=f"Average document sentences", leave=False):
        tot_sentences += len(data["sentences"])
    return tot_sentences


def documentsTokens(dataset):
    tot_tokens = 0
    for data in tqdm(dataset, desc=f"Average document tokens", leave=False):
        for sentence in data["sentences"]:
            tot_tokens += len(re.findall(r'\w+', sentence))
    return tot_tokens


def referenceSummariesTokens(dataset):
    tot_tokens = 0
    for data in tqdm(dataset, desc=f"Average reference summary tokens", leave=False):
        tot_tokens += len(re.findall(r'\w+', data["ref_summary"]))
    return tot_tokens


def extractiveSummariesTokens(dataset):
    tot_tokens = 0
    for data in tqdm(dataset, desc=f"Average extractive summary tokens", leave=False):
        for i, selected in enumerate(data["labels"]):
            if selected: tot_tokens += len(re.findall(r'\w+', data["sentences"][i]))
    return tot_tokens


def extractiveSummariesSentences(dataset):
    tot_sentences = 0
    for data in tqdm(dataset, desc=f"Average extractive summary sentences", leave=False):
        tot_sentences += data["labels"].count(True)
    return tot_sentences