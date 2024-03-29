import spacy
import spacy_universal_sentence_encoder
import pytest

from . import utils


def test_load_from_pipe_default_en():
    nlp = spacy.blank("en")
    p = nlp.add_pipe("universal_sentence_encoder")
    assert p != None
    utils._test_default_text(nlp)


@pytest.mark.skipif(not utils.multi, reason="no multi")
def test_load_from_pipe_default_xx():
    nlp = spacy.blank("xx")
    p = nlp.add_pipe("universal_sentence_encoder")
    assert p != None
    utils._test_default_text(nlp)


def test_load_from_module_default_en():
    nlp = spacy_universal_sentence_encoder.load_model("en_use_md")
    assert nlp != None
    utils._test_default_text(nlp)


@pytest.mark.skipif(not utils.multi, reason="no multi")
def test_load_from_module_default_xx():
    nlp = spacy_universal_sentence_encoder.load_model("xx_use_md")
    assert nlp != None
    utils._test_default_text(nlp)


@pytest.mark.skipif(not utils.full_test, reason="not full test")
def test_load_models_from_pipe_all():
    for model_name in utils.model_names:
        if not utils.multi and model_name.startswith("xx"):
            # skip multi-language models
            continue
        nlp = spacy.blank("en")
        p = nlp.add_pipe(
            "universal_sentence_encoder", config={"model_name": model_name}
        )
        assert p != None
        utils._test_default_text(nlp)


@pytest.mark.skipif(not utils.full_test or not utils.multi, reason="not full test")
def test_load_models_from_module_all():
    for model_name in utils.model_names:
        if not utils.multi and model_name.startswith("xx"):
            # skip multi-language models
            continue
        nlp = spacy_universal_sentence_encoder.load_model(model_name)
        utils._test_default_text(nlp)
