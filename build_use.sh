set -e

# create the nlp and save to disk
python create.py
# overwrite meta.json
cp meta/meta.json use_model/meta.json

# create the package
mkdir -p use_package
python -m spacy package use_model use_package --force
pushd use_package/en_use-0.1.3
# zip it
python setup.py sdist
# install the tar.gz from dist/en_use-0.1.1.tar.gz
pip install dist/en_use-0.1.3.tar.gz
popd