# Spacy - Universal Sentence Encoder

Make use of Google's Universal Sentence Encoder directly within SpaCy.
This library lets you embed [Docs](https://spacy.io/api/doc), [Spans](https://spacy.io/api/span) and [Tokens](https://spacy.io/api/token) from the [Universal Sentence Encoder family available on TensorFlow Hub](https://tfhub.dev/google/collections/universal-sentence-encoder/1).

For using sentence-BERT in spaCy, see https://github.com/MartinoMensio/spacy-sentence-bert

## Motivation
There are many different reasons to not always use BERT. For example to have embeddings that are tuned specifically for another task (e.g. sentence similarity). See this very useful blog article:
https://blog.floydhub.com/when-the-best-nlp-model-is-not-the-best-choice/

The Universal Sentence Encoder is trained on different tasks which are more suited to identifying sentence similarity. [Google AI blog](https://ai.googleblog.com/2018/05/advances-in-semantic-textual-similarity.html) [paper](https://arxiv.org/abs/1803.11175)

This library uses the [`user_hooks` of spaCy](https://spacy.io/usage/processing-pipelines#custom-components-user-hooks) to use an external model for the vectors, in this case a simple wrapper to the models available on TensorFlow Hub.

## Install

You can install this library from:
- github: `pip install git+https://github.com/MartinoMensio/spacy-universal-sentence-encoder.git`
- pyPI: `pip install spacy-universal-sentence-encoder`

Or you can install the following pre-packaged models with pip:

| model name | source | pip package |
|------------|--------|---|
| en_use_md  | https://tfhub.dev/google/universal-sentence-encoder | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder/releases/download/v0.3.2/en_use_md-0.3.2.tar.gz#en_use_md-0.3.2` |
| en_use_lg  | https://tfhub.dev/google/universal-sentence-encoder-large | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder/releases/download/v0.3.2/en_use_lg-0.3.2.tar.gz#en_use_lg-0.3.2` |
| xx_use_md  | https://tfhub.dev/google/universal-sentence-encoder-multilingual | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder/releases/download/v0.3.2/xx_use_md-0.3.2.tar.gz#xx_use_md-0.3.2` |
| xx_use_lg  | https://tfhub.dev/google/universal-sentence-encoder-multilingual-large | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder/releases/download/v0.3.2/xx_use_lg-0.3.2.tar.gz#xx_use_lg-0.3.2` |


## Usage

First you have to import your model.

If you installed the model standalone packages (see table above) you can use the usual spacy API to load this model:

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
import spacy_universal_sentence_encoder
# this is your nlp object that can be any spaCy model
nlp = spacy.load('en_core_web_sm')


# get the pipe component
use_overwrite_vectors = nlp.create_pipe('use_overwrite_vectors')
# add to your nlp the pipeline stage
nlp.add_pipe(use_overwrite_vectors)
# use the vector with the default `en_use_md` model
doc = nlp('Hi')

# extend the nlp pipeline with the `en_use_lg` model
spacy_universal_sentence_encoder.create_from(nlp, 'en_use_lg')
```

## Common issues

Here you can find the most common issues with possible solutions.

### Using a pre-downloaded model

If you want to use a model that you have already downloaded from TensorFlow Hub, belonging to the [Universal Sentence Encoder family](https://tfhub.dev/google/collections/universal-sentence-encoder/1), you can use it by doing the following:

- locate the full path of the folder where you have downloaded and extracted the model. Let's suppose the location is `/Users/foo/Downloads`
- rename the folder of the extracted model (the one directly containing the folders `variables` and the file `saved_model.pb`) to the sha1 hash of the TFHub model [source](https://medium.com/@xianbao.qian/how-to-run-tf-hub-locally-without-internet-connection-4506b850a915). The mapping URL / sha1 values is the following:
  - [`en_use_md`](https://tfhub.dev/google/universal-sentence-encoder/4): `063d866c06683311b44b4992fd46003be952409c`
  - [`en_use_lg`](https://tfhub.dev/google/universal-sentence-encoder-large/5): `c9fe785512ca4a1b179831acb18a0c6bfba603dd`
  - [`xx_use_md`](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3): `26c892ffbc8d7b032f5a95f316e2841ed4f1608c`
  - [`xx_use_lg`](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3): `97e68b633b7cf018904eb965602b92c9f3ad14c9`
- set the environment variable `TFHUB_CACHE_DIR` to the location containing the renamed folder, in our case `/Users/foo/Downloads` (set it before trying to download the model)
- Now load your model and it should see that it was already downloaded

### Serialisation

To serialise and deserialise nlp objects, SpaCy does not restore `user_hooks` after deserialisation, so a call to `from_bytes` will result in not using the TensorFlow vectors, so the similarities won't be good. For this reason the suggested solution is:

- serialise with `bytes = doc.to_bytes()` normally
- deserialise with `spacy_universal_sentence_encoder.doc_from_bytes(nlp, bytes)` which will also restore the user hooks

### Multiprocessing

This library, relying on TensorFlow, is not fork-safe. This means that if you are using this library inside multiple processes (e.g. with a `multiprocessing.pool.Pool`), your processes will deadlock.
The solutions are:
- use a thread-based environment (e.g. `multiprocessing.pool.ThreadPool`)
- only use this library inside the created processes (first create the processes and then import and use the library)

## Utils

To build and upload
```bash
VERSION=0.3.2
# build the standalone models (17)
./build_models.sh
# build the archive at dist/spacy_universal_sentence_encoder-${VERSION}.tar.gz
python setup.py sdist
# upload to pypi
twine upload dist/spacy_universal_sentence_encoder-${VERSION}.tar.gz
```