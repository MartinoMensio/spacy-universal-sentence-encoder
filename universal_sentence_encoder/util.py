try:  # Python 3.8
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # noqa: F401


pkg_meta = importlib_metadata.metadata(__name__.split(".")[0])

configs = {
    'en_use_md': {
        'spacy_base_model': 'en_core_web_sm',
        'tfhub_model_url': 'https://tfhub.dev/google/universal-sentence-encoder/4'
    },
    'en_use_lg': {
        'spacy_base_model': 'en_core_web_sm',
        'tfhub_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5'
    },
    'en_use_sm': { # NOT WORKING: TypeError: 'AutoTrackable' object is not callable
        'spacy_base_model': 'en_core_web_sm',
        'tfhub_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-lite/2'
    },
    'xx_use_md': {
        'spacy_base_model': 'xx_ent_wiki_sm',
        'tfhub_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
    },
    'xx_use_lg': {
        'spacy_base_model': 'xx_ent_wiki_sm',
        'tfhub_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'
    },
    'xx_use_8lang': { # NOT WORKING: tensorflow.python.framework.errors_impl.NotFoundError: Op type not registered 'SentencepieceEncodeSparse'
        'spacy_base_model': 'xx_ent_wiki_sm',
        'tfhub_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-xling-many/1'
    },
    'xx_use_en_de': { # NOT WORKING: tensorflow.python.framework.errors_impl.NotFoundError: Op type not registered 'SentencepieceEncodeSparse'
        'spacy_base_model': 'xx_ent_wiki_sm',
        'tfhub_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-xling-many/1'
    },
    'xx_use_en_es': { # NOT WORKING: tensorflow.python.framework.errors_impl.NotFoundError: Op type not registered 'SentencepieceEncodeSparse'
        'spacy_base_model': 'xx_ent_wiki_sm',
        'tfhub_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-xling/en-es/1'
    },
    'xx_use_en_fr': { # NOT WORKING: tensorflow.python.framework.errors_impl.NotFoundError: Op type not registered 'SentencepieceEncodeSparse'
        'spacy_base_model': 'xx_ent_wiki_sm',
        'tfhub_model_url': 'https://tfhub.dev/google/universal-sentence-encoder-xling/en-fr/1'
    },
}