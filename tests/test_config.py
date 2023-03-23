"""
Tests for the config parameters
"""
import spacy
import numpy as np
import spacy_universal_sentence_encoder

from . import utils


def test_config_default_en():
    nlp = spacy.blank("en")
    p = nlp.add_pipe("universal_sentence_encoder")
    assert p != None
    utils._test_default_text(nlp)


def test_use_model_url():
    nlp = spacy.blank("en")
    url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
    url2 = "https://tfhub.dev/google/universal-sentence-encoder/4"
    p = nlp.add_pipe("universal_sentence_encoder", config={"use_model_url": url})
    assert p != None
    doc = utils._test_default_text(nlp)
    vec = doc.vector
    # TODO: consistency of attribute name
    assert p.model_url == url
    # TODO: should not be changeable, or needs to reload model
    # p.model_url = url2
    # doc2 = utils._test_default_text(nlp)
    # assert p.use_model_url == url2
    # similarity = doc.similarity(doc2)
    # assert similarity < 0.95
    nlp2 = spacy.blank("en")
    p2 = nlp2.add_pipe("universal_sentence_encoder", config={"use_model_url": url2})
    assert p2 != None
    doc2 = utils._test_default_text(nlp2)
    vec2 = doc2.vector
    assert p2.model_url == url2
    # now check that they are different, as a proof that two different models were used
    # similarity = doc.similarity(doc2)
    cos_sim = np.dot(vec, vec2) / (np.linalg.norm(vec) * np.linalg.norm(vec2))
    assert cos_sim < 0.95


def test_preprocessor_url():
    nlp = spacy.blank("en")
    url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    url2 = "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2"
    p = nlp.add_pipe("universal_sentence_encoder", config={"preprocessor_url": url})
    assert p != None
    doc = utils._test_default_text(nlp)
    vec = doc.vector
    assert p.preprocessor_url == url
    # TODO: should not be changeable, or needs to reload model
    # p.preprocessor_url = url2
    # doc2 = utils._test_default_text(nlp)
    # assert p.preprocessor_url == url2
    # similarity = doc.similarity(doc2)
    # assert similarity < 0.95
    nlp2 = spacy.blank("en")
    p2 = nlp2.add_pipe("universal_sentence_encoder", config={"preprocessor_url": url2})
    assert p2 != None
    doc2 = utils._test_default_text(nlp2)
    vec2 = doc2.vector
    assert p2.preprocessor_url == url2
    # now check that they are very similar still, because preprocessor doesn't change much
    # similarity = doc.similarity(doc2)
    cos_sim = np.dot(vec, vec2) / (np.linalg.norm(vec) * np.linalg.norm(vec2))
    assert cos_sim > 0.95


def test_model_name():
    nlp = spacy.blank("en")
    p = nlp.add_pipe("universal_sentence_encoder", config={"model_name": "en_use_lg"})
    assert p != None
    doc = utils._test_default_text(nlp)
    vec = doc.vector
    # assert p.model_name == "en_use_lg"
    # TODO: should not be changeable, or needs to reload model
    # p.model_name = "en_use_md"
    # doc2 = utils._test_default_text(nlp)
    # assert p.model_name == "en_use_md"
    # similarity = doc.similarity(doc2)
    # assert similarity < 0.95
    nlp2 = spacy.blank("en")
    p2 = nlp2.add_pipe("universal_sentence_encoder", config={"model_name": "xx_use_md"})
    assert p2 != None
    doc2 = utils._test_default_text(nlp2)
    vec2 = doc2.vector
    # assert p2.model_name == "en_use_md"
    # now check that they are different, as a proof that two different models were used
    # similarity = doc.similarity(doc2)
    cos_sim = np.dot(vec, vec2) / (np.linalg.norm(vec) * np.linalg.norm(vec2))
    assert cos_sim < 0.95


def test_config_cache():
    nlp = spacy.blank("en")
    p = nlp.add_pipe("universal_sentence_encoder", config={"enable_cache": False})
    assert p != None
    utils._test_default_text(nlp)
    assert p.enable_cache == False
    # TODO: should not be changeable, or needs to reload model
    p.enable_cache = True
    utils._test_default_text(nlp)
    assert p.enable_cache == True
    nlp = spacy.blank("en")
    p = nlp.add_pipe("universal_sentence_encoder", config={"enable_cache": True})
    assert p != None
    utils._test_default_text(nlp)
    assert p.enable_cache == True
    # TODO: should not be changeable, or needs to reload model
    p.enable_cache = False
    utils._test_default_text(nlp)
    assert p.enable_cache == False


def test_debug():
    nlp = spacy.blank("en")
    p = nlp.add_pipe("universal_sentence_encoder", config={"debug": True})
    assert p != None
    utils._test_default_text(nlp)
    assert p.debug == True
    # TODO: should not be changeable, or needs to reload model
    p.debug = False
    utils._test_default_text(nlp)
    assert p.debug == False
    nlp = spacy.blank("en")
    p = nlp.add_pipe("universal_sentence_encoder", config={"debug": False})
    assert p != None
    utils._test_default_text(nlp)
    assert p.debug == False
    # TODO: should not be changeable, or needs to reload model
    p.debug = True
    utils._test_default_text(nlp)
    assert p.debug == True
