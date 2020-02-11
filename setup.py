from setuptools import setup, find_packages
# to import the version and also run the module one time (download cache model)
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
        ],
        # keep the models folder in the wheel: the tfhub cache is shipped during the installation
        package_data={"universal_sentence_encoder": ["models/*", "models/*/*", "models/*/*/*"]}
    )

if __name__ == "__main__":
    setup_package()