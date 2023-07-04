import numpy as np


"""
    Keeps track of metrics and their averages.
"""
class MetricsLogger():
    def __init__(self):
        self.reset()


    def reset(self):
        self.total_loss = []
        self.total_recall = []
        self.total_rouge1 = { "fmeasure": [], "precision": [], "recall": [] }
        self.total_rouge2 = { "fmeasure": [], "precision": [], "recall": [] }
        self.total_rougeL = { "fmeasure": [], "precision": [], "recall": [] }


    def add(self, type, value):
        if type == "loss":
            self.total_loss.append(value)
        elif type == "recall":
            self.total_recall.append(value)
        elif type == "rouge":
            self.total_rouge1["fmeasure"].append(value["rouge1"]["fmeasure"])
            self.total_rouge1["precision"].append(value["rouge1"]["precision"])
            self.total_rouge1["recall"].append(value["rouge1"]["recall"])
            self.total_rouge2["fmeasure"].append(value["rouge2"]["fmeasure"])
            self.total_rouge2["precision"].append(value["rouge2"]["precision"])
            self.total_rouge2["recall"].append(value["rouge2"]["recall"])
            self.total_rougeL["fmeasure"].append(value["rougeL"]["fmeasure"])
            self.total_rougeL["precision"].append(value["rougeL"]["precision"])
            self.total_rougeL["recall"].append(value["rougeL"]["recall"])


    def averages(self):
        return {
            "loss": np.average(self.total_loss),
            "recall": np.average(self.total_recall),
            "rouge1": {
                "recall": np.average(self.total_rouge1["recall"]),
                "precision": np.average(self.total_rouge1["precision"]),
                "fmeasure": np.average(self.total_rouge1["fmeasure"])
            },
            "rouge2": {
                "recall": np.average(self.total_rouge2["recall"]),
                "precision": np.average(self.total_rouge2["precision"]),
                "fmeasure": np.average(self.total_rouge2["fmeasure"])
            },
            "rougeL": {
                "recall": np.average(self.total_rougeL["recall"]),
                "precision": np.average(self.total_rougeL["precision"]),
                "fmeasure": np.average(self.total_rougeL["fmeasure"])
            }
        }


    def format(self, types):
        out = "| "
        avgs = self.averages()

        if "loss" in types:
            out += f"Loss {avgs['loss']:.5f} | "
        if "recall" in types:
            out += f"Recall {avgs['recall']:.5f} | "
        if "rouge" in types:
            out += f"ROUGE-1 r: {avgs['rouge1']['recall']*100:.2f} -- p: {avgs['rouge1']['precision']*100:.2f} -- f1: {avgs['rouge1']['fmeasure']*100:.2f} | "
            out += f"ROUGE-2 r: {avgs['rouge2']['recall']*100:.2f} -- p: {avgs['rouge2']['precision']*100:.2f} -- f1: {avgs['rouge2']['fmeasure']*100:.2f} | "
            out += f"ROUGE-L r: {avgs['rougeL']['recall']*100:.2f} -- p: {avgs['rougeL']['precision']*100:.2f} -- f1: {avgs['rougeL']['fmeasure']*100:.2f} | "

        return out