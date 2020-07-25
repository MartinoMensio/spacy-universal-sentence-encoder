# coding: utf8
from __future__ import unicode_literals

import os
from pathlib import Path
import spacy
from spacy.util import load_model_from_init_py, get_model_meta
from spacy.language import Language
from spacy.tokens import Span
from spacy.matcher import Matcher
import warnings
from . import util

from .util import create_lang as load_model

__version__ = util.pkg_meta["version"]

from .language import UniversalSentenceEncoder, TFHubWrapper
UniversalSentenceEncoder.install_extensions()

# warning suppress for empty vocabulary
warnings.filterwarnings('ignore', message=r"\[W007\]", category=UserWarning)

# language factories for the pipeline stages
Language.factories['use_add_model_to_doc'] = lambda nlp, **cfg: AddModelToDoc(nlp, **cfg)
Language.factories['use_overwrite_vectors'] = lambda nlp, **cfg: OverwriteVectors(nlp, **cfg)

def load(**overrides):
    return load_model_from_init_py(__file__, **overrides)

def create_from(nlp, model_name):
    '''From an existing `nlp` object, adds the vectors from the specific `model_name` by adding pipeline stages'''
    if model_name not in util.configs:
        raise ValueError(f'Model "{model_name}" not available')
    config = util.configs[model_name]
    return UniversalSentenceEncoder.create_nlp(config, nlp)

class AddModelToDoc(object):
    '''Factory of the `use_add_model_to_doc` pipeline stage. It tells spacy how to create the stage '''
    name = 'use_add_model_to_doc'

    def __init__(self, nlp):
        # TODO handle cache flag
        model_name = f'{nlp.meta["lang"]}_{nlp.meta["name"]}'
        cfg = util.configs[model_name]
        self.model = TFHubWrapper(cfg['use_model_url'], enable_cache=True)

    def __call__(self, doc):
        doc._.use_model = self.model

        return doc


class OverwriteVectors(object):
    '''Factory of the `use_overwrite_vectors` pipeline stage. It tells spacy how to create the stage '''
    name = 'use_overwrite_vectors'

    def __init__(self, nlp):
        pass

    def __call__(self, doc):
        UniversalSentenceEncoder.overwrite_vectors(doc)

        return doc