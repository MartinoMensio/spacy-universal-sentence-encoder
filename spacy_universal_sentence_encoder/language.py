from typing import Dict, Any
from spacy.pipeline import Pipe
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.util import get_lang_class
from absl import logging
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import os
import pathlib
import spacy

class UniversalSentenceEncoder(Language):

    @staticmethod
    def install_extensions():
        def get_encoding(token_span_doc):
            return token_span_doc.doc._.use_model.embed_one(token_span_doc)
        
        # placeholder for the model
        Doc.set_extension('use_model', default=None, force=True)
        # set the extension on doc, span and token
        Token.set_extension('universal_sentence_encoding', getter=get_encoding, force=True)
        Span.set_extension('universal_sentence_encoding', getter=get_encoding, force=True)
        Doc.set_extension('universal_sentence_encoding', getter=get_encoding, force=True)

    @staticmethod
    def overwrite_vectors(doc):
        # https://spacy.io/usage/processing-pipelines#custom-components-user-hooks
        doc.user_hooks["vector"] = lambda a: a._.universal_sentence_encoding
        doc.user_span_hooks["vector"] = lambda a: a._.universal_sentence_encoding
        doc.user_token_hooks["vector"] = lambda a: a._.universal_sentence_encoding
        
        return doc


    @staticmethod
    def create_nlp(cfg, nlp=None):
        model = TFHubWrapper(cfg['use_model_url'], enable_cache=True)

        def add_model_to_doc(doc):
            doc._.use_model = model
            return doc
        
        if not nlp:
            nlp = spacy.blank(cfg['spacy_base_model'])
            nlp.add_pipe(nlp.create_pipe('sentencizer'))
        nlp.add_pipe(add_model_to_doc, name='use_add_model_to_doc', first=True)
        nlp.add_pipe(UniversalSentenceEncoder.overwrite_vectors, name='use_overwrite_vectors', after='use_add_model_to_doc')
        return nlp





class TFHubWrapper(object):
    embed_cache: Dict[str, Any]
    enable_cache: bool
    model_url: str
    model: Any


    def __init__(self, use_model_url='https://tfhub.dev/google/universal-sentence-encoder/4', enable_cache=True):
        self.embed_cache = {}

        logging.set_verbosity(logging.ERROR)
        self.model_url = use_model_url
        self.enable_cache = enable_cache
        # models saved here
        os.environ['TFHUB_CACHE_DIR'] = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / 'models')
        # show download info
        os.environ['TFHUB_DOWNLOAD_PROGRESS'] = '1'
        self.model = hub.load(self.model_url)
        # print(f'module {self.model_url} loaded')

    def embed(self, texts):
        # print('embed called')
        result = self.model(texts)
        result = np.array(result)
        return result


    # extension implementation
    def embed_one(self, span):
        text = span.text
        # print('enable_cache', TFHubWrapper.enable_cache)
        if self.enable_cache and text in self.embed_cache:
            return self.embed_cache[text]
        else:
            result = self.embed([text])[0]
            if self.enable_cache:
                self.embed_cache[text] = result
            return result
