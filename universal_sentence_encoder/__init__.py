# coding: utf8
from __future__ import unicode_literals

import os
import spacy
from pathlib import Path
from spacy.util import load_model_from_init_py, get_model_meta
from spacy.language import Language
from spacy.tokens import Span
from spacy.matcher import Matcher

from .language import UniversalSentenceEncoder
UniversalSentenceEncoder.install_extensions()

# warning suppress for empty vocabulary 
# (setting on the environ wouldn't work if spacy is already loaded)
spacy.errors.SPACY_WARNING_IGNORE.append('W007')

Language.factories['overwrite_vectors'] = lambda nlp, **cfg: OverwriteVectors(nlp, **cfg)

def load(**overrides):
    return load_model_from_init_py(__file__, **overrides)

class OverwriteVectors(object):
    name = "overwrite_vectors"

    def __init__(self, nlp, enable_cache):
        # enable_cache = cfg.get('enable_cache', True)
        # UniversalSentenceEncoder.install_extensions()
        print('enable_cache', enable_cache)
        UniversalSentenceEncoder.tf_wrapper.enable_cache = enable_cache

    def __call__(self, doc):
        UniversalSentenceEncoder.overwrite_vectors(doc)

        return doc