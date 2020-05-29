# Spacy - Universal Sentence Encoder

## Motivation
Motivation to have different models:
https://blog.floydhub.com/when-the-best-nlp-model-is-not-the-best-choice/
The USE is trained on different tasks which are more suited to identifying sentence similarity. Source Google AI blog https://ai.googleblog.com/2018/05/advances-in-semantic-textual-similarity.html 

## Install

You can install this repository:
- pyPI: `pip install spacy-universal-sentence-encoder`
- github: `pip install git+https://https://github.com/MartinoMensio/spacy-universal-sentence-encoder-tfhub`

Or you can install the following pre-packaged models with pip:

| model name | source | pip package |
|------------|--------|---|
| en_use_md  | https://tfhub.dev/google/universal-sentence-encoder | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder-tfhub/releases/download/en_use_md-0.2.1/en_use_md-1.tar.gz#en_use_md-0.2.1 ` |
| en_use_lg  | https://tfhub.dev/google/universal-sentence-encoder-large | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder-tfhub/releases/download/en_use_lg-0.2.1/en_use_lg-0.2.1.tar.gz#en_use_lg-0.2.1` |
| xx_use_md  | https://tfhub.dev/google/universal-sentence-encoder-multilingual | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder-tfhub/releases/download/xx_use_md-0.2.1/xx_use_md-0.2.1.tar.gz#xx_use_md-0.2.1 ` |
| xx_use_lg  | https://tfhub.dev/google/universal-sentence-encoder-multilingual-large | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder-tfhub/releases/download/xx_use_lg-0.2.1/xx_use_lg-0.2.1.tar.gz#xx_use_lg-0.2.1` |


## Usage

First you have to import your model.

If you installed the model packages (see table above) you can use the usual spacy API to load this model:

```python
import spacy
nlp = spacy.load('en_use_md')
```

Otherwise you need to load the model in the following way (the first time that it is run, it downloads the model)
```python
import spacy_universal_sentence_encoder
nlp = spacy_universal_sentence_encoder.load_model('xx_use_lg')
```

Then you can use the models

```python
# get two documents
doc_1 = nlp('Hi there, how are you?')
doc_2 = nlp('Hello there, how are you doing today?')
# get the vector of the Doc, Span or Token
print(doc_1.vector.shape)
print(doc_1[3].vector.shape)
print(doc_1[2:4].vector.shape)
# or use the similarity method that is based on the vectors, on Doc, Span or Token
print(doc_1.similarity(doc_2[0:7]))
```

You can use the model on a already available language pipeline (e.g. to keep your components or to have better parsing than the base spacy model used here):

```python
import spacy
# this is your nlp object that can be anything
nlp = spacy.load('en_core_web_sm')
# import the specific

# get the pipe component
overwrite_vectors = nlp.create_pipe('overwrite_vectors')
# add to your nlp the pipeline stage
nlp.add_pipe(overwrite_vectors)
# use the vector with the default `en_use_md` model
doc = nlp('Hi')


# or use a different model
other_model_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'

# by setting the extension `tfhub_model_url` on the doc
doc._.tfhub_model_url = other_module_url

# or by adding a pipeline component that sets on every document
def set_tfhub_model_url(doc):
    doc._.tfhub_model_url = other_model_url
    return doc

# add this pipeline component before the `overwrite_vectors`, because it will look at that extension
nlp.add_pipe(set_tfhub_model_url, before='overwrite_vectors')

```
