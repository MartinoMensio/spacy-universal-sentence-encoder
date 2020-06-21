# Spacy - Universal Sentence Encoder

Make use of Google's Universal Sentence Encoder directly within SpaCy.
This library lets you embed [Docs](https://spacy.io/api/doc), [Spans](https://spacy.io/api/span) and [Tokens](https://spacy.io/api/token) from the [Universal Sentence Encoder family available on TensorFlow Hub](https://tfhub.dev/google/collections/universal-sentence-encoder/1).

## Motivation
There are many different reasons to not always use BERT. For example to have embeddings that are tuned specifically for another task (e.g. sentence similarity). See this very useful blog article:
https://blog.floydhub.com/when-the-best-nlp-model-is-not-the-best-choice/

The Universal Sentence Encoder is trained on different tasks which are more suited to identifying sentence similarity. [Google AI blog](https://ai.googleblog.com/2018/05/advances-in-semantic-textual-similarity.html) [paper](https://arxiv.org/abs/1803.11175)

This library uses the [`user_hooks` of spaCy](https://spacy.io/usage/processing-pipelines#custom-components-user-hooks) to use an external model for the vectors, in this case a simple wrapper to the models available on TensorFlow Hub.

## Install

You can install this library from:
- github: `pip install git+https://github.com/MartinoMensio/spacy-universal-sentence-encoder-tfhub.git`
- pyPI: `pip install spacy-universal-sentence-encoder` **not working** (problems with base models dependency xx_ent_wiki_sm and en_core_web_sm: base models cannot be specified as dependencies for pyPI, better to keep releases on GitHub)

Or you can install the following pre-packaged models with pip:

| model name | source | pip package |
|------------|--------|---|
| en_use_md  | https://tfhub.dev/google/universal-sentence-encoder | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder-tfhub/releases/download/en_use_md-0.2.3/en_use_md-0.2.3.tar.gz#en_use_md-0.2.3` |
| en_use_lg  | https://tfhub.dev/google/universal-sentence-encoder-large | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder-tfhub/releases/download/en_use_lg-0.2.3/en_use_lg-0.2.3.tar.gz#en_use_lg-0.2.3` |
| xx_use_md  | https://tfhub.dev/google/universal-sentence-encoder-multilingual | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder-tfhub/releases/download/xx_use_md-0.2.3/xx_use_md-0.2.3.tar.gz#xx_use_md-0.2.3` |
| xx_use_lg  | https://tfhub.dev/google/universal-sentence-encoder-multilingual-large | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder-tfhub/releases/download/xx_use_lg-0.2.3/xx_use_lg-0.2.3.tar.gz#xx_use_lg-0.2.3` |


## Usage

First you have to import your model.

If you installed the model packages (see table above) you can use the usual spacy API to load this model:

```python
import spacy
nlp = spacy.load('en_use_md')
```

Otherwise you need to load the model in the following way (the first time that it is run, it downloads the model):

```python
import spacy_universal_sentence_encoder
nlp = spacy_universal_sentence_encoder.load_model('xx_use_lg')
```

Then you can use the models:

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

You can use the model on an already available language pipeline (e.g. to integrate with your custom components or to have better parsing than the base spaCy model used here):

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
doc._.tfhub_model_url = other_model_url

# or by adding a pipeline component that sets on every document
def set_tfhub_model_url(doc):
    doc._.tfhub_model_url = other_model_url
    return doc

# add this pipeline component before the `overwrite_vectors`, because it will look at that extension
nlp.add_pipe(set_tfhub_model_url, before='overwrite_vectors')

```
