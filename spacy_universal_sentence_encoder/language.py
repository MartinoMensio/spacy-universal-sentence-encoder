from typing import Dict, Any
from spacy.pipeline import Pipe
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.util import get_lang_class
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
# TODO error on MacOS https://github.com/google/sentencepiece/issues/309
# import sentencepiece 
# import tf_sentencepiece
import numpy as np
import os
import pathlib
import spacy

class UniversalSentenceEncoder(Language):
    tf_wrapper: Any

    @staticmethod
    def install_extensions():
        def get_encoding(token_span_doc):
            tfhub_model_url = token_span_doc.doc._.tfhub_model_url
            wrapper = TFHubWrapper.get_instance(tfhub_model_url)
            return wrapper.embed_one(token_span_doc)
        
        # set the extension both on doc and span level
        # Token.set_extension('universal_sentence_encoding', getter=UniversalSentenceEncoder.tf_wrapper.embed_one, force=True)
        # Span.set_extension('universal_sentence_encoding', getter=UniversalSentenceEncoder.tf_wrapper.embed_one, force=True)
        # Doc.set_extension('universal_sentence_encoding', getter=UniversalSentenceEncoder.tf_wrapper.embed_one, force=True)
        Doc.set_extension('tfhub_model_url', default='https://tfhub.dev/google/universal-sentence-encoder/4', force=True)
        Token.set_extension('universal_sentence_encoding', getter=get_encoding, force=True)
        Span.set_extension('universal_sentence_encoding', getter=get_encoding, force=True)
        Doc.set_extension('universal_sentence_encoding', getter=get_encoding, force=True)

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
    def create_nlp(spacy_base_model, tfhub_model_url):
        # nlp = spacy.blank(language_base)
        # nlp.add_pipe(nlp.create_pipe('sentencizer'))

        def save_tfhub_model_url(doc):
            doc._.tfhub_model_url = tfhub_model_url
            return doc
        
        nlp = spacy.load(spacy_base_model)
        nlp.add_pipe(save_tfhub_model_url)
        nlp.add_pipe(UniversalSentenceEncoder.overwrite_vectors)
        return nlp


    @staticmethod
    def create_wrapper(tfhub_model_url, enable_cache=True):
        """Helper method, run to do the loading now"""
        UniversalSentenceEncoder.tf_wrapper = TFHubWrapper.get_instance(tfhub_model_url)
        # TODO the enable_cache with singleton is not a great idea
        UniversalSentenceEncoder.tf_wrapper.enable_cache = enable_cache



class TFHubWrapper(object):
    embed_cache: Dict[str, Any]
    enable_cache = True
    instances = {}

    @staticmethod
    def get_instance(tfhub_model_url):
        # singleton
        if tfhub_model_url not in TFHubWrapper.instances:
            instance = TFHubWrapper(tfhub_model_url)
            TFHubWrapper.instances[tfhub_model_url] = instance
        else:
            instance = TFHubWrapper.instances[tfhub_model_url]
        return instance


    def __init__(self, tfhub_model_url):
        self.embed_cache = {}

        logging.set_verbosity(logging.ERROR)
        self.module_url = tfhub_model_url
        # models saved here
        os.environ['TFHUB_CACHE_DIR'] = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / 'models')
        self.model = hub.load(self.module_url)
        # print(f'module {self.module_url} loaded')

    def embed(self, texts):
        # print('embed called')
        result = self.model(texts)
        result = np.array(result)
        return result


    # extension implementation
    def embed_one(self, span):
        text = span.text
        # print(self.module_url, span)
        # print('enable_cache', TFHubWrapper.enable_cache)
        if TFHubWrapper.enable_cache and text in self.embed_cache:
            # print('already cached')
            return self.embed_cache[text]
        else:
            result = self.embed([text])[0]
            if TFHubWrapper.enable_cache:
                self.embed_cache[text] = result
            return result
