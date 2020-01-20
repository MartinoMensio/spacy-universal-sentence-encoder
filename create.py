from universal_sentence_encoder import language

nlp = language.UniversalSentenceEncoder.create_nlp()
print(nlp.pipe_names)
doc = nlp('Hello my friend')
print(doc.vector)
nlp.to_disk('use_model')
