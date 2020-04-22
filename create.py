import typer
from universal_sentence_encoder import language, util

def main(model_name):
    if model_name not in util.configs:
        raise ValueError(f'Model "{model_name}" not available')
    selected_config = util.configs[model_name]
    nlp = language.UniversalSentenceEncoder.create_nlp(selected_config['spacy_base_model'], selected_config['tfhub_model_url'])
    # nlp.vocab.reset_vectors(width=0)
    print(nlp.pipe_names)
    doc = nlp('Hello my friend')
    print(doc.vector.shape)
    nlp.to_disk(f'models/{model_name}')

if __name__ == "__main__":
    typer.run(main)