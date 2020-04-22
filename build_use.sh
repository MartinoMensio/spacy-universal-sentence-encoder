set -e

# select here which one to build

# MODEL_NAME='en_use_md'
# MODEL_NAME='en_use_lg'
# MODEL_NAME='xx_use_md'
MODEL_NAME='xx_use_lg'

mkdir -p models/$MODEL_NAME
# create the nlp and save to disk
python create.py $MODEL_NAME
# overwrite meta.json
cp meta/$MODEL_NAME.json models/$MODEL_NAME/meta.json

# create the package
mkdir -p packages
python -m spacy package models/$MODEL_NAME packages --force
pushd packages/$MODEL_NAME-0.2.0
# zip it
python setup.py sdist
# install the tar.gz from dist/en_use-0.1.1.tar.gz
pip install dist/$MODEL_NAME-0.2.0.tar.gz
popd