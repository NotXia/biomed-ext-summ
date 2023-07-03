import torch.nn as nn
from utilities.summary import select
import spacy



class BaseSummarizer(nn.Module):
    def __init__(self, model_name, model_family, input_size):
        super().__init__()
        self._tokenizer = None
        self._doc2sentences = None
        self.model_family = model_family
        self.model_name = model_name
        self.input_size = input_size

    @property
    def tokenizer(self):
        # Tokenizer initialized lazily
        if self._tokenizer is None:
            self._tokenizer = self._createTokenizer()
        return self._tokenizer


    def _createTokenizer(self):
        raise NotImplementedError


    def forward(self, batch):
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
        raise NotImplementedError
    
    
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
        raise NotImplementedError


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