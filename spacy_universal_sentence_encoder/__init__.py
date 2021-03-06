# coding: utf8
from __future__ import unicode_literals

import os
from pathlib import Path
import spacy
from spacy.util import load_model_from_init_py, get_model_meta
from spacy.language import Language
from spacy.tokens import Span, Doc
from spacy.matcher import Matcher
import warnings
from . import util

from .util import create_lang as load_model

__version__ = util.pkg_meta["version"]

from . import language

# warning suppress for empty vocabulary
warnings.filterwarnings('ignore', message=r"\[W007\]", category=UserWarning)

def load(**overrides):
    return load_model_from_init_py(__file__, **overrides)

def create_from(nlp, use_model_code):
    '''From an existing `nlp` object, adds the vectors from the specific `use_model_code` by adding pipeline stages'''
    if use_model_code not in util.configs:
        raise ValueError(f'Model "{use_model_code}" not available')
    config = util.configs[use_model_code]
    return language.create_nlp(config, nlp)

def doc_from_bytes(nlp, bytes):
    """Returns a serialised doc from the bytes coming from `doc.to_bytes()` """
    doc = Doc(nlp.vocab).from_bytes(bytes)
    language.set_hooks(doc)
    return doc
