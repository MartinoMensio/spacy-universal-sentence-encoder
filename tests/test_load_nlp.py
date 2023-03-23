import spacy


def test_basic_load():
    nlp = spacy.blank("en")
    p = nlp.add_pipe("universal_sentence_encoder")
    assert p != None
    assert "universal_sentence_encoder" in nlp.pipe_names


def test_load_vector():
    nlp = spacy.blank("en")
    p = nlp.add_pipe("universal_sentence_encoder")
    assert p != None
    assert "universal_sentence_encoder" in nlp.pipe_names
    doc = nlp("This is a test")
    vector = doc.vector
    assert vector is not None
    shape = vector.shape
    assert shape != None
