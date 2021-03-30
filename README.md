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

Compatibility:
- python 3.6/3.7/3.8 (constraint from [tensorflow](https://pypi.org/project/tensorflow/))
- tensorflow>=2.4.0,<3.0.0
- spacy>=3.0.0,<4.0.0 (SpaCy v3 API changed a lot from v2)

To use the multilingual version of the models, you need to install the extra named `multi` with the command: `pip install spacy-universal-sentence-encoder[multi]`. This installs the dependency `tensorflow-text` that is required to run the multilingual models. Note that this library is still not available for Windows operating systems (https://github.com/tensorflow/text/issues/291).

In alternative, you can install the following standalone pre-packaged models with pip. The same limitation for multilingual models applies (when trying to install a multilingual model on Windows, pip will say that no tensorflow-text is available). Each model can be installed independently:

| model name | source | pip package |
|------------|--------|---|
| en_use_md  | https://tfhub.dev/google/universal-sentence-encoder | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder/releases/download/v0.4.1/en_use_md-0.4.1.tar.gz#en_use_md-0.4.1` |
| en_use_lg  | https://tfhub.dev/google/universal-sentence-encoder-large | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder/releases/download/v0.4.1/en_use_lg-0.4.1.tar.gz#en_use_lg-0.4.1` |
| xx_use_md  | https://tfhub.dev/google/universal-sentence-encoder-multilingual | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder/releases/download/v0.4.1/xx_use_md-0.4.1.tar.gz#xx_use_md-0.4.1` |
| xx_use_lg  | https://tfhub.dev/google/universal-sentence-encoder-multilingual-large | `pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder/releases/download/v0.4.1/xx_use_lg-0.4.1.tar.gz#xx_use_lg-0.4.1` |


## Usage

### Loading the model

If you installed the model standalone packages (see table above) you can use the usual spacy API to load this model:

```python
import spacy
nlp = spacy.load('en_use_md')
```

Otherwise you need to load the model in the following way:

```python
import spacy_universal_sentence_encoder
nlp = spacy_universal_sentence_encoder.load_model('xx_use_lg')
```

The third option is to load the model on your existing spaCy pipeline:

```python
import spacy
# this is your nlp object that can be any spaCy model
nlp = spacy.load('en_core_web_sm')

# add the pipeline stage (will be mapped to the most adequate model from the table above, en_use_md)
nlp.add_pipe('universal_sentence_encoder')
```

In all of the three options, the first time that you load a certain Universal Sentence Encoder model, it will be downloaded from TF Hub (see section below to use an already downloaded model, or to change the location of the model files).

The last option (using `nlp.add_pipe`) can be customised with the following configurations:

- `use_model_url`: allows to use a specific TFHub URL
- `model_name`: to load a specific model instead of mapping the current (language, size) to one of the options in the table above
- `enable_cache`: default `True`, enables an internal cache to avoid embedding the same text (doc/span/token) twice. It makes the computation faster (when enough duplicates are embedded) but has a memory footprint because all the embeddings extracted are kept in the cache
- `debug`: default `False` shows debugging information.

To use the configurations, when adding the pipe stage pass a dict as additional argument, for example:

```python
nlp.add_pipe('universal_sentence_encoder', config={'enable_cache': False})
```

### Use the embeddings

After adding to the pipeline, you can use the embedding models by using the various properties and methods of Docs, Spans and Tokens:

```python
# load as before
import spacy
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('universal_sentence_encoder')

# get two documents
doc_1 = nlp('Hi there, how are you?')
doc_2 = nlp('Hello there, how are you doing today?')
# Inspect the shape of the Doc, Span and Token vectors
print(doc_1.vector.shape) # the full document representation
print(doc_1[3], doc_1[3].vector.shape) # the word "how"
print(doc_1[3:6], doc_1[3:6].vector.shape) # the span "how are you"

# or use the similarity method that is based on the vectors, on Doc, Span or Token
print(doc_1.similarity(doc_2[0:7]))
```

## Common issues

Here you can find the most common issues with possible solutions.

### Using a pre-downloaded model

If you want to use a model that you have already downloaded from TensorFlow Hub, belonging to the [Universal Sentence Encoder family](https://tfhub.dev/google/collections/universal-sentence-encoder/1), you can use it by doing the following:

- locate the full path of the folder where you have downloaded and extracted the model. Let's suppose the location is `$HOME/tfhub_models`
- rename the folder of the extracted model (the one directly containing the folders `variables` and the file `saved_model.pb`) to the sha1 hash of the TFHub model [source](https://medium.com/@xianbao.qian/how-to-run-tf-hub-locally-without-internet-connection-4506b850a915). The mapping URL / sha1 values is the following:
  - [`en_use_md`](https://tfhub.dev/google/universal-sentence-encoder/4): `063d866c06683311b44b4992fd46003be952409c`
  - [`en_use_lg`](https://tfhub.dev/google/universal-sentence-encoder-large/5): `c9fe785512ca4a1b179831acb18a0c6bfba603dd`
  - [`xx_use_md`](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3): `26c892ffbc8d7b032f5a95f316e2841ed4f1608c`
  - [`xx_use_lg`](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3): `97e68b633b7cf018904eb965602b92c9f3ad14c9`
- set the environment variable `TFHUB_CACHE_DIR` to the location containing the renamed folder, for example `$HOME/tfhub_models` (set it before trying to download the model: `export TFHUB_CACHE_DIR=$HOME/tfhub_models`)
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
# change version
VERSION=0.4.1
# change version references everywhere
# update locally installed package
pip install -r requirements.txt
# build the standalone models (17)
./build_models.sh
# build the archive at dist/spacy_universal_sentence_encoder-${VERSION}.tar.gz
python setup.py sdist
# upload to pypi
twine upload dist/spacy_universal_sentence_encoder-${VERSION}.tar.gz
# upload language packages to github
```