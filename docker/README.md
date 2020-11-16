# Docker instructions

## Build the image
```bash
docker build -t use .
```

## Create the container

You can use pre-downloaded models from [TFHub](https://tfhub.dev/google/collections/universal-sentence-encoder/1) by using volume mapping.
If you want to do so, at the same time you need to provide the environment variable `TFHUB_CACHE_DIR` which will be used by the library to find the downloaded models.

```bash
docker run -it --rm -v `pwd`/models/universal_sentence_encoder:/SOME_SIMPLE_PATH_HERE -e TFHUB_CACHE_DIR=/SOME_SIMPLE_PATH_HERE --entrypoint /bin/bash nlp
```
