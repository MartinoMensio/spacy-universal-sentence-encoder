import json
from pathlib import Path

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

def create_lang(model_name, download_now=True):
    from . import language
    if model_name not in configs:
        raise ValueError(f'Model "{model_name}" not available')
    selected_config = configs[model_name]
    nlp = language.UniversalSentenceEncoder.create_nlp(selected_config['spacy_base_model'], selected_config['tfhub_model_url'])
    with open(Path(__file__).parent.absolute() / 'meta' / f'{model_name}.json') as f:
        nlp.meta = json.load(f)
    if download_now:
        doc = nlp('Test')
        # this line makes the TF Hub download the model now
        _ = doc.vector.shape
    return nlp