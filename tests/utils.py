import os

model_names = [
    "en_use_md",
    "en_use_lg",
    "xx_use_md",
    "xx_use_lg",
    "en_use_cmlm_md",
    "en_use_cmlm_lg",
    "xx_use_cmlm",
    "xx_use_cmlm_br",
]

full_test = os.environ.get("FULL_TEST", False)


def _test_default_text(nlp):
    assert "universal_sentence_encoder" in nlp.pipe_names
    doc = nlp("This is a test")
    vector = doc.vector
    assert vector is not None
    shape = vector.shape
    assert shape != None
    return doc


def _test_default_similarity_identical(nlp):
    doc1 = nlp("This is a test")
    doc2 = nlp("This is a test")
    similarity = doc1.similarity(doc2)
    assert similarity is not None
    assert similarity > 0.95
    return similarity


def _test_default_similarity_different(nlp):
    doc1 = nlp("This is a test")
    doc2 = nlp("This is something else")
    similarity = doc1.similarity(doc2)
    assert similarity is not None
    assert similarity < 0.95
    return similarity
