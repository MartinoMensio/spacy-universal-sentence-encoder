# coding: utf8
from __future__ import unicode_literals

from pathlib import Path
from spacy.util import load_model_from_init_py, get_model_meta
from spacy.language import Language
from spacy.tokens import Span
from spacy.matcher import Matcher

from .language import UniversalSentenceEncoder

__version__ = get_model_meta(Path(__file__).parent)['version']

Language.factories['overwrite_vectors'] = lambda nlp, **cfg: OverwriteVectors(nlp, **cfg)

def load(**overrides):
    return load_model_from_init_py(__file__, **overrides)

class OverwriteVectors(object):
    name = "overwrite_vectors"

    def __init__(self, nlp, **cfg):
        UniversalSentenceEncoder.install_extensions()

    def __call__(self, doc):
        UniversalSentenceEncoder.overwrite_vectors(doc)

        return doc