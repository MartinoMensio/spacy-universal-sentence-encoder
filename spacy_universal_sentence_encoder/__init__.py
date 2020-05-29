# coding: utf8
from __future__ import unicode_literals

import os
from pathlib import Path
import spacy
from spacy.util import load_model_from_init_py, get_model_meta
from spacy.language import Language
from spacy.tokens import Span
from spacy.matcher import Matcher
from . import util
from .util import create_lang as load_model

__version__ = util.pkg_meta["version"]

from .language import UniversalSentenceEncoder
UniversalSentenceEncoder.install_extensions()

# warning suppress for empty vocabulary 
# (setting on the environ wouldn't work if spacy is already loaded)
spacy.errors.SPACY_WARNING_IGNORE.append('W007')

Language.factories['save_tfhub_model_url'] = lambda nlp, **cfg: SaveTfhubModelUrl(nlp, **cfg)
Language.factories['overwrite_vectors'] = lambda nlp, **cfg: OverwriteVectors(nlp, **cfg)

def load(**overrides):
    return load_model_from_init_py(__file__, **overrides)

class SaveTfhubModelUrl(object):
    name = 'save_tfhub_model_url'

    def __init__(self, nlp):
        model_name = f'{nlp.meta["lang"]}_{nlp.meta["name"]}'
        # print('init.model_name', model_name)
        self.tfhub_model_url = util.configs[model_name]['tfhub_model_url']
        # load tfhub now (not compulsory but nice to have it loaded when running `spacy.load`)
        UniversalSentenceEncoder.create_wrapper(self.tfhub_model_url)

    def __call__(self, doc):
        # print('SaveTfhubModelUrl called. tfhub_model_url =', self.tfhub_model_url)
        doc._.tfhub_model_url = self.tfhub_model_url

        return doc


class OverwriteVectors(object):
    name = "overwrite_vectors"

    def __init__(self, nlp):
        # enable_cache = cfg.get('enable_cache', True)
        # UniversalSentenceEncoder.install_extensions()
        # print('enable_cache', enable_cache)
        # print(nlp.meta)
        # load tfhub now (not compulsory but nice to have it loaded when running `spacy.load`)
        #UniversalSentenceEncoder.create_wrapper(enable_cache=enable_cache)
        pass

    def __call__(self, doc):
        UniversalSentenceEncoder.overwrite_vectors(doc)

        return doc