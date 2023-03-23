import spacy
import spacy_universal_sentence_encoder
import pytest

from . import utils


def test_similarity_identical_default_en():
    nlp = spacy_universal_sentence_encoder.load_model("en_use_md")
    utils._test_default_similarity_identical(nlp)


def test_similarity_different_default_en():
    nlp = spacy_universal_sentence_encoder.load_model("en_use_md")
    utils._test_default_similarity_different(nlp)


def test_similarity_identical_default_xx():
    nlp = spacy_universal_sentence_encoder.load_model("xx_use_md")
    utils._test_default_similarity_identical(nlp)


def test_similarity_different_default_xx():
    nlp = spacy_universal_sentence_encoder.load_model("xx_use_md")
    utils._test_default_similarity_different(nlp)


@pytest.mark.skipif(not utils.full_test, reason="not full test")
def test_similarity_identical_all():
    for model_name in utils.model_names:
        nlp = spacy_universal_sentence_encoder.load_model(model_name)
        utils._test_default_similarity_identical(nlp)


@pytest.mark.skipif(not utils.full_test, reason="not full test")
def test_similarity_different_all():
    for model_name in utils.model_names:
        nlp = spacy_universal_sentence_encoder.load_model(model_name)
        utils._test_default_similarity_different(nlp)
