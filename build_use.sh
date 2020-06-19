set -e

# select here which one to build

# MODEL_NAME=''
# MODEL_NAME=''
# MODEL_NAME=''
for MODEL_NAME in en_use_md en_use_lg xx_use_md xx_use_lg
do
    mkdir -p models/$MODEL_NAME
    # create the nlp and save to disk
    python create.py $MODEL_NAME
    # overwrite meta.json
    cp spacy_universal_sentence_encoder/meta/$MODEL_NAME.json models/$MODEL_NAME/meta.json

    # create the package
    mkdir -p packages
    python -m spacy package models/$MODEL_NAME packages --force
    pushd packages/$MODEL_NAME-0.2.3
    # zip it
    python setup.py sdist
    # install the tar.gz from dist folder
    pip install dist/$MODEL_NAME-0.2.3.tar.gz
    popd
done
