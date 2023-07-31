# Biomedical extractive summarization

An evaluation of general-domain and biomedical pretrained language models for biomedical extractive summarization.


## Available weights
Fine tuned weights for the following models are available on Hugging Face:
| Model | Hugging Face |
|-------|--------------|
| Longformer | [NotXia/longformer-bio-ext-summ](https://huggingface.co/NotXia/longformer-bio-ext-summ) |
| PubMedBERT | [NotXia/pubmedbert-bio-ext-summ](https://huggingface.co/NotXia/pubmedbert-bio-ext-summ) |

## Installation
Move into `src` and run:
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```


## Preprocessing
A dataset has to be first converted into an extractive form.\
Available datasets are: `cnn_dailymail`, `ms2`, `cochrane` and `sumpubmed`.
```
python abs2ext.py                               \
        --dataset=<dataset name>                \
        --output=<path to output directory>     \
        --proc=<number of processes to create>
```

Then, an extractive dataset has to be tokenized for a specific model.\
Supported model families are: `BERT`, `RoBERTa`, `DistilBERT`, `MobileBERT` and `Longformer`.
```
python preprocess.py                            \
    --dataset-dir=<path to extractive dataset>  \
    --model=<HuggingFace model to tokenize for> \
    --output=<path to output directory>         \
    --proc=<number of processes to create>
```


## Training
To train a model run:
```
python train.py                                             \
    --model=<HuggingFace model>                             \
    --dataset=<path to tokenized dataset>                   \
    --epochs=<training epochs>                              \
    --lr=<learning rate>                                    \
    --batch-size=<batch size>                               \
    --accum-steps=<gradient accumulation steps>             \
    --checkpoint-best                                       \
    --checkpoints-freq=<number of epochs>                   \
    --checkpoints-path=<path where checkpoints are saved>   \
    --history-path=<path where training history is saved>
```
If mixed precision is needed, add `--mixed-precision`.


## Evaluating
To evaluate a trained model, run:

```
python eval.py                                      \
    --checkpoint=<path to model checkpoint>         \
    --dataset=<path to extractive dataset>          \
    --splits=<comma separated splits>               \
    --strategy-count=<number of selected sentences>
```
Other summarization strategy can be used instead of `strategy-count`.\
Run `python eval.py -h` for more details.

To evaluate using the Oracle, run:
```
python eval.py                              \
    --checkpoint=<path to model checkpoint> \
    --dataset=<path to extractive dataset>  \
    --splits=<comma separated splits>       \
    --oracle
```

To evaluate using LEAD-N, run:
```
python eval.py                              \
    --checkpoint=<path to model checkpoint> \
    --dataset=<path to extractive dataset>  \
    --splits=<comma separated splits>       \
    --lead=<number of sentences>
```
