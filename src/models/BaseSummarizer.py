import torch
from utilities.summary import select
import spacy
import itertools
from utilities.summary import splitDocument



class BaseSummarizer(torch.nn.Module):
    def __init__(self, model_name, input_size):
        super().__init__()
        self._doc2sentences = None
        self.model_name = model_name
        self.input_size = input_size
        self.tokenizer = None


    def forward(self, batch):
        raise NotImplementedError


    """ 
        Makes a prediction on a tokenized document.
        It is assumed that the document is within the maximum input size.

        Parameters
        ----------
            chunk_tokens : str[]
                Tokens of the document.

        Returns
        -------
            predictions : Tensor<float>
                Score for each sentence.
    """
    def predictChunk(self, chunk_tokens):
        raise NotImplementedError


    """ 
        Summarize a document from the dataset.
        Useful during validation.

        Parameters
        ----------
            predictions : Tensor<float>
                Score for each sentence.

            doc_ids : Tensor<int>
                Document already converted to ids.

            summary_size : int
                Number of sentences to select.

        Returns
        -------
            selected_sentences : str[]
                Sentences part of the summary (already decoded).

            selected_indexes : int[]
                Indexes of the selected sentences.
    """
    def summarizeFromDataset(self, predictions, doc_ids, summary_size):
        doc_ids = [id for id in doc_ids if id != self.tokenizer.pad_token_id]
        doc_sentences = [self.tokenizer.decode(list(ids)[:-1]) 
                            for x, ids in itertools.groupby(doc_ids, lambda id: id == self.tokenizer.cls_token_id) if not x]
        return self.summarizeSentences(doc_sentences, "count", summary_size, predictions=predictions)
    

    """
        Make a prediction on a list of sentences.

        Parameters
        ----------
            sentences : str[]
                Sentences of the document.

        Returns
        -------
            predictions : Tensor<float>
                Score for each sentence.
    """
    def predict(self, sentences):
        doc_tokens = self.tokenizer.tokenize( f"{self.tokenizer.sep_token}{self.tokenizer.cls_token}".join(sentences) )
        doc_tokens = [self.tokenizer.cls_token] + doc_tokens + [self.tokenizer.sep_token]
        # If the document is too long, it is split and processed separately. 
        # The resulting predictions are then concatenated.
        doc_chunks = splitDocument(doc_tokens, self.tokenizer.cls_token, self.tokenizer.sep_token, self.input_size)
        predictions = torch.as_tensor([]).to(next(self.parameters()).device)

        for chunk in doc_chunks:
            chunk_preds = self.predictChunk(chunk)
            predictions = torch.cat((predictions, chunk_preds))

        return predictions


    """
        Create a summary of a list of sentences with a given strategy.

        Parameters
        ----------
            sentences : str[]
                List of sentences of a document to summarize.

            strategy : str
                Strategy to summarize the document:
                - 'length': summary with a maximum length (strategy_args is the maximum length).
                - 'count': summary with the given number of sentences (strategy_args is the number of sentences).
                - 'ratio': summary proportional to the length of the document (strategy_args is the ratio [0, 1]).
                - 'threshold': summary only with sentences with a score higher than a given value (strategy_args is minimum score).

            strategy_args : any
                Parameters of the strategy.

            predictions : Tensor<float>|None
                Score for each sentence. If None, it will be computed internally.
        Returns
        -------
            selected_sentences : str[]
                Sentences part of the summary.

            selected_indexes : int[]
                Indexes of the selected sentences.
    """
    def summarizeSentences(self, sentences, strategy="ratio", strategy_args=0.3, predictions=None):
        if predictions is None:
            predictions = self.predict(sentences)
        return select(sentences, predictions, strategy, strategy_args)


    """
        Create a summary of a document with a given strategy.

        Parameters
        ----------
            document : str
                Document to summarize.

            strategy : str
                Same as `summarizeSentences`

            strategy_args : any
                Parameters of the strategy.
        Returns
        -------
            selected_sentences : str[]
                Sentences part of the summary.

            selected_indexes : int[]
                Indexes of the selected sentences.
    """
    def summarize(self, document, strategy="ratio", strategy_args=0.3):
        if self._doc2sentences is None: 
            self._doc2sentences = spacy.load("en_core_web_sm")
        
        doc_sentences = [sent.text for sent in self._doc2sentences(document).sents]
        return self.summarizeSentences(doc_sentences, strategy, strategy_args)