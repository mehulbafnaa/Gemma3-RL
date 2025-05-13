#!/bin/bash
# Export your Kaggle username and API key
export KAGGLE_USERNAME=your_username

export KAGGLE_KEY=your_API_key

curl -L -u $KAGGLE_USERNAME:$KAGGLE_KEY\
  -o ./model.tar.gz\
  https://www.kaggle.com/api/v1/models/google/gemma-3/flax/gemma3-4b-it/1/download


