# Spacy - Universal Sentence Encoder

## Motivation
Motivation to have different models:
https://blog.floydhub.com/when-the-best-nlp-model-is-not-the-best-choice/
The USE is trained on different tasks which are more suited to identifying sentence similarity. Source Google AI blog https://ai.googleblog.com/2018/05/advances-in-semantic-textual-similarity.html 

## Build model
Or use the model built provided in the "packages" of this repo.
```bash
bash build_use.sh
```

## Install

```bash
pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder-tfhub/releases/download/en_use-0.1.0/en_use-0.1.0.tar.gz#en_use-0.1.0
```

## Usage

```
import spacy
nlp = spacy.load('en_use')
```

## TODOs 

Model config:
- `use_cache` flag
