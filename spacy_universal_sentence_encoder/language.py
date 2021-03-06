from typing import Dict, Any, List
from spacy.pipeline import Pipe
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.util import get_lang_class
from absl import logging
import tensorflow_hub as hub
try:
    # installed with the extra `multi`
    import tensorflow_text
except:
    # without extra `multi`
    pass
import numpy as np
import os
import pathlib
import spacy

from . import util

# magic
def get_vector(token_span_doc):
    doc = token_span_doc.doc
    use_model_url = doc._.use_model_url
    # if not use_model_url:
    model = UniversalSentenceEncoder.get_model(use_model_url)
    vector = model.embed_one(token_span_doc)
    return vector


# install/register the extensions
Doc.set_extension('use_model_url', default=None, force=True)
Token.set_extension('universal_sentence_encoding', getter=get_vector, force=True)
Span.set_extension('universal_sentence_encoding', getter=get_vector, force=True)
Doc.set_extension('universal_sentence_encoding', getter=get_vector, force=True)

# the pipeline stage factory
@Language.factory('universal_sentence_encoder', default_config={
    'use_model_url': None,
    'model_name': None,
    'enable_cache': True,
    'debug': False
})
def use_model_factory(nlp, name, use_model_url, model_name, enable_cache, debug):
    if debug:
        print('use_model_factory:', nlp, 'use_model_url', use_model_url, 'model_name', model_name)
    if use_model_url:
        config = None
        # prioritize custom use_model_url
        model_url = use_model_url
    elif model_name:
        # user has chosen from the configs available in utils.configs
        if model_name not in util.configs:
            raise ValueError(f'Parameter {model_name} must be one of {util.configs.keys()}')
        config = util.configs[model_name]
        model_url = config['use_model_url']
    else:
        # inherit from nlp object
        # the language code needs to match
        meta_lang = nlp.meta['lang']
        # try to map from the model name
        meta_name = nlp.meta["name"]
        model_name = f'{meta_lang}_{meta_name}'
        if not model_name in util.configs:
            if meta_name.endswith('_lg'):
                best = 'use_lg'
            else:
                best = 'use_md'
            model_name_best = f'{meta_lang}_{best}'
            if not model_name_best in util.configs:
                raise ValueError(f"Couldn't map nlp.meta['lang']={meta_lang} and nlp.meta['name']={meta_name} to one of the models.\n"\
                    f"Please provide the parameter 'model_name' as one of {list(util.configs.keys())}")
            else:
                if debug:
                    print(f'Your model was mapped to {model_name_best}')
                    model_name = model_name_best
        config = util.configs[model_name]
        model_url = config['use_model_url']

    if debug:
        print('model_url=', model_url)

    if config and config['spacy_base_model'] == 'xx':
        # double check `multi` extra is installed
        if not 'tensorflow_text' in globals():
            raise ValueError('This multilanguage model requires tensorflow_text. Install it with: pip install spacy-universal-sentence-encoder[multi]')

    
    model = UniversalSentenceEncoder(model_url, enable_cache, debug)
    return model

class UniversalSentenceEncoder(object):

    models: Dict[str, Any] = {}

    working_pid = None

    def __init__(self, model_url, enable_cache=True, debug=False):
        
        self.model_url = model_url
        self.enable_cache=enable_cache
        self.debug=debug
        # load it now so that when the extension getter will call it, the model will be already loaded
        _ = UniversalSentenceEncoder.get_model(self.model_url, self.enable_cache, self.debug)


    def __call__(self, doc):
        doc._.use_model_url = self.model_url
        set_hooks(doc)

        return doc

    @staticmethod
    def get_model(use_model_url, enable_cache=True, debug=False):
        # print('getting', use_model_url)

        # PID checking: TensorFlow gets stuck with multiple processes
        my_pid = os.getpid()
        if UniversalSentenceEncoder.working_pid:
            if my_pid != UniversalSentenceEncoder.working_pid:
                print('WARNING: this model won\'t be able to be executed because TensorFlow is not fork-safe.\n'
                    'Your processes will be stuck soon.\n\n'
                    'Use threads instead of processes or load this library in the process directly!')
        else:
            UniversalSentenceEncoder.working_pid = my_pid

        
        if use_model_url in UniversalSentenceEncoder.models:
            # print('model in cache')
            model = UniversalSentenceEncoder.models[use_model_url] 
        else:
            # print('model not in cache')
            model = TFHubWrapper(use_model_url, enable_cache=enable_cache, debug=debug)
            UniversalSentenceEncoder.models[use_model_url] = model
        return model


def set_hooks(doc):
    """Places in the doc.vector, span.vector and token.vector the values from the extension"""
    # https://spacy.io/usage/processing-pipelines#custom-components-user-hooks
    doc.user_hooks["vector"] = lambda a: a._.universal_sentence_encoding
    doc.user_span_hooks["vector"] = lambda a: a._.universal_sentence_encoding
    doc.user_token_hooks["vector"] = lambda a: a._.universal_sentence_encoding
    
    return doc


def create_nlp(cfg, nlp=None):
    spacy_base_model = cfg['spacy_base_model']
    use_model_url = cfg['use_model_url']
    
    if not nlp:
        nlp = spacy.blank(spacy_base_model)
        nlp.add_pipe('sentencizer')
    nlp.add_pipe('universal_sentence_encoder', config={'use_model_url': use_model_url})
    return nlp


class TFHubWrapper(object):
    embed_cache: Dict[str, Any]
    enable_cache: bool
    model_url: str
    model: Any


    def __init__(self, use_model_url, enable_cache=True, debug=False):
        self.embed_cache = {}

        logging.set_verbosity(logging.ERROR)
        self.model_url = use_model_url
        self.enable_cache = enable_cache
        # models saved here
        if not os.environ.get('TFHUB_CACHE_DIR'):
            os.environ['TFHUB_CACHE_DIR'] = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / 'models')
        # show download info
        os.environ['TFHUB_DOWNLOAD_PROGRESS'] = '1'
        self.model = hub.load(self.model_url)
        if not self.model:
            raise ValueError(f'Impossible to load model with use_model_url={use_model_url}')
        if debug:
            print(f'module {self.model_url} loaded')


    def embed(self, texts: List[str]):
        """Embed multiple texts"""
        # print('embed TFHubWrapper called')
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
