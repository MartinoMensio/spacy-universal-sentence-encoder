import spacy

def test_basic_load():
    nlp = spacy.blank("en")
    p = nlp.add_pipe('universal_sentence_encoder')
    assert p != None
    assert 'universal_sentence_encoder' in nlp.pipe_names