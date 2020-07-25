import json
from pathlib import Path

try:  # Python 3.8
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # noqa: F401


pkg_meta = importlib_metadata.metadata(__name__.split(".")[0])

configs = {
    'en_use_md': {
        'spacy_base_model': 'en',
        'use_model_url': 'https://tfhub.dev/google/universal-sentence-encoder/4'
    },
    'en_use_lg': {
        'spacy_base_model': 'en',
        'use_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5'
    },
    'en_use_sm': { # NOT WORKING: Requires working with tf1 compat mode https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder_lite.ipynb
        'spacy_base_model': 'en',
        'use_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-lite/2'
    },
    'xx_use_md': {
        'spacy_base_model': 'xx',
        'use_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
    },
    'xx_use_lg': {
        'spacy_base_model': 'xx',
        'use_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'
    },
    'xx_use_8lang': { # NOT WORKING: tensorflow.python.framework.errors_impl.NotFoundError: Op type not registered 'SentencepieceEncodeSparse'
        'spacy_base_model': 'xx',
        'use_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-xling-many/1'
    },
    'xx_use_en_de': { # NOT WORKING: tensorflow.python.framework.errors_impl.NotFoundError: Op type not registered 'SentencepieceEncodeSparse'
        'spacy_base_model': 'xx',
        'use_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-xling-many/1'
    },
    'xx_use_en_es': { # NOT WORKING: tensorflow.python.framework.errors_impl.NotFoundError: Op type not registered 'SentencepieceEncodeSparse'
        'spacy_base_model': 'xx',
        'use_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-xling/en-es/1'
    },
    'xx_use_en_fr': { # NOT WORKING: tensorflow.python.framework.errors_impl.NotFoundError: Op type not registered 'SentencepieceEncodeSparse'
        'spacy_base_model': 'xx',
        'use_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-xling/en-fr/1'
    },
}

def create_lang(model_name):
    from . import language
    if model_name not in configs:
        raise ValueError(f'Model "{model_name}" not available')
    selected_config = configs[model_name]
    nlp = language.UniversalSentenceEncoder.create_nlp(selected_config)
    with open(Path(__file__).parent.absolute() / 'meta' / f'{model_name}.json') as f:
        nlp.meta = json.load(f)

    return nlp