import torch
from models.BaseSummarizer import BaseSummarizer
from utilities.transformer import TransformerInterEncoder
from utilities.utilities import padToSize
import itertools



class T5Summarizer(BaseSummarizer):
    def __init__(self, t5_model, t5_tokenizer, input_size=512):
        super().__init__(t5_model.name_or_path, input_size)
        self.t5 = t5_model
        self.tokenizer = t5_tokenizer
        self.interSentenceEncoder = TransformerInterEncoder(self.t5.config.hidden_size)


    def forward(self, batch):
        document_ids = batch["ids"].to(self.t5.device)
        clss_mask = batch["clss_mask"].to(self.t5.device)
        attn_mask = batch["attn_mask"].to(self.t5.device)

        tokens_out = self.t5(input_ids=document_ids, attention_mask=attn_mask, return_dict=False)
        tokens_out = tokens_out[0]

        out = []
        logits_out = []

        for i in range(len(tokens_out)): # Batch handling
            clss_out = tokens_out[i][clss_mask[i], :]
            sentences_scores, logits = self.interSentenceEncoder(clss_out)
            padding = torch.zeros(self.input_size - sentences_scores.shape[0]).to(sentences_scores.device)

            out.append( torch.cat((sentences_scores, padding)) )
            logits_out.append( torch.cat((logits, padding)) )

        return torch.stack(out), torch.stack(logits_out)


    def predictChunk(self, chunk_tokens):
        doc_ids = self.tokenizer.convert_tokens_to_ids(chunk_tokens)
        clss_mask = [True if token == self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token) else False for token in doc_ids]
        attn_mask = [1 for _ in range(len(doc_ids))]

        # Simulates a batch of size 1
        batch = {}
        
        batch["ids"] = torch.as_tensor( [padToSize(doc_ids, 512, self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token))] ).to(self.t5.device)
        batch["clss_mask"] = torch.as_tensor( [padToSize(clss_mask, 512, False)] ).to(self.t5.device)
        batch["attn_mask"] = torch.as_tensor( [padToSize(attn_mask, 512, 0)] ).to(self.t5.device)

        self.eval()
        with torch.no_grad():
            predictions, _ = self(batch)
            return predictions[0][:chunk_tokens.count(self.tokenizer.eos_token)]

    def predict(self, sentences):
        doc_tokens = self.tokenizer.tokenize( f"{self.tokenizer.eos_token}".join(sentences) )
        doc_tokens = doc_tokens + [self.tokenizer.eos_token]
        
        # If the document is too long, it is split and processed separately.
        # The resulting predictions are then concatenated.
        doc_chunks = splitDocument(doc_tokens, None, self.tokenizer.eos_token, self.input_size)
        
        predictions = torch.as_tensor([]).to(next(self.parameters()).device)

        for chunk in doc_chunks:
            chunk_preds = self.predictChunk(chunk)
            predictions = torch.cat((predictions, chunk_preds))
            
        return predictions
        

    def summarizeFromDataset(self, predictions, doc_ids, summary_size):
        doc_ids = [id for id in doc_ids if id != self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]
        doc_sentences = [self.tokenizer.decode(list(ids)[:-1])
                            for x, ids in itertools.groupby(doc_ids, lambda id: id == self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)) if not x]
        return self.summarizeSentences(doc_sentences, "count", summary_size, predictions=predictions)