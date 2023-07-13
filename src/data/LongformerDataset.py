import torch
from utilities.utilities import padToSize
from tqdm import tqdm


class LongformerDataset(torch.utils.data.Dataset):
    def __init__(self, ext_dataset, tokenizer, input_size=4096):
        self.labels = []
        self.documents = []

        for data in tqdm(ext_dataset, desc="Loading dataset"):
            labels = [1 if label else 0 for label in data["labels"]]
            ids = data["doc_ids"]
            clss_mask = [False for _ in range(input_size)]
            for i in data["cls_idxs"]: clss_mask[i] = True
            attn_mask = [1 for _ in range(len(data["doc_ids"]))]
            global_attn_mask = [0 for _ in range(input_size)]
            for i in data["cls_idxs"]: global_attn_mask[i] = 1

            self.labels.append( torch.tensor(padToSize(labels, input_size, 0)) ) 
            self.documents.append({
                "ids":                  torch.tensor( padToSize(ids, input_size, tokenizer.pad_token_id) ),
                "clss_mask":            torch.tensor( clss_mask ),
                "attn_mask":            torch.tensor( padToSize(attn_mask, input_size, 0) ).unsqueeze(0),
                "global_attn_mask":     torch.tensor( global_attn_mask ).unsqueeze(0),
                "ref_summary":  data["ref_summary"],
                "num_sentences": len(data["labels"])
            })

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx], self.labels[idx]