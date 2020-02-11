from setuptools import setup, find_packages
import universal_sentence_encoder

def setup_package():
    setup(
        name="universal_sentence_encoder",
        entry_points={
            "spacy_factories": ["overwrite_vectors = universal_sentence_encoder:OverwriteVectors"]
        },
        version=universal_sentence_encoder.__version__,
        packages=find_packages(),
        install_requires =[
            'tensorflow==2.1.0',
            'spacy',
            'tensorflow-hub',
            'seaborn'
        ]
    )

if __name__ == "__main__":
    setup_package()