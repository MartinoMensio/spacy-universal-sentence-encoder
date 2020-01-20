# Spacy - Universal Sentence Encoder

## Motivation
Motivation to have different models:
https://blog.floydhub.com/when-the-best-nlp-model-is-not-the-best-choice/
The USE is trained on different tasks which are more suited to identifying sentence similarity. Source Google AI blog https://ai.googleblog.com/2018/05/advances-in-semantic-textual-similarity.html 

## Install

Create model folder: `python create.py`
run setup.py: `python setup.py develop` or `pip install -e .`

## Usage

```
import spacy
nlp = spacy.load('universal_sentence_encoder_model')
```

## TODOs 

Model config:
- `use_cache` flag

## Commands old


docker run -p 8501:8501 -v `pwd`/tfserving/universal_encoder:/models/universal_encoder -e MODEL_NAME=universal_encoder -t tensorflow/serving:1.13.1

run: python create.py

check status: http://localhost:8501/v1/models/universal_encoder

POST: http://localhost:8501/v1/models/universal_encoder:predict

{
	"instances": [{
		"text": "Is there anyone there?"
	}]
}

result:
{
    "predictions": [
        [EMBEDDED_SENTENCE_VALUE]
    ]
}