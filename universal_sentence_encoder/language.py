from typing import Dict, Any
from spacy.pipeline import Pipe
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.util import get_lang_class
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import spacy

class UniversalSentenceEncoder(Language):
    tf_wrapper: Any

    @staticmethod
    def install_extensions():
        UniversalSentenceEncoder.tf_wrapper = TFHubWrapper()
        # set the extension both on doc and span level
        Span.set_extension('universal_sentence_encoding', getter=UniversalSentenceEncoder.tf_wrapper.embed_one, force=True)
        Doc.set_extension('universal_sentence_encoding', getter=UniversalSentenceEncoder.tf_wrapper.embed_one, force=True)

    @staticmethod
    def overwrite_vectors(doc):
        # https://spacy.io/usage/processing-pipelines#custom-components-user-hooks
        # doc.user_hooks["similarity"] = lambda a, b: similarity(a, b)
        # doc.user_span_hooks["similarity"] = lambda a, b: similarity(a, b)
        # doc.user_token_hooks["similarity"] = lambda a, b: similarity(a, b)
        doc.user_hooks["vector"] = lambda a: a._.universal_sentence_encoding
        doc.user_span_hooks["vector"] = lambda a: a._.universal_sentence_encoding
        doc.user_token_hooks["vector"] = lambda a: a._.universal_sentence_encoding
        # doc.user_hooks["vector_norm"] = lambda a: a._.universal_sentence_encoding
        # doc.user_span_hooks["vector_norm"] = lambda a: a._.universal_sentence_encoding
        # doc.user_token_hooks["vector_norm"] = lambda a: a._.universal_sentence_encoding
        return doc


    @staticmethod
    def create_nlp(language_base='en'):
        nlp = spacy.blank(language_base)
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        nlp.add_pipe(UniversalSentenceEncoder.overwrite_vectors)
        return nlp

class UniversalSentenceEncoderPipe(Pipe):
    pass


class TFHubWrapper(object):
    embed_cache: Dict[str, Any]
    enable_cache = True

    def __init__(self):
        self.embed_cache = {}

        logging.set_verbosity(logging.ERROR)
        self.module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
        self.model = hub.load(self.module_url)
        print("module %s loaded" % self.module_url)

    def embed(self, texts):
        # print('embed called')
        result = self.model(texts)
        result = np.array(result)
        return result


    # extension implementation
    def embed_one(self, span):
        text = span.text
        print('enable_cache', TFHubWrapper.enable_cache)
        if TFHubWrapper.enable_cache and text in self.embed_cache:
            return self.embed_cache[text]
        else:
            result = self.embed([text])[0]
            if TFHubWrapper.enable_cache:
                self.embed_cache[text] = result
            return result
