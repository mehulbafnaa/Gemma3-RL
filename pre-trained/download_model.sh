#!/bin/bash
# Export your Kaggle username and API key
export KAGGLE_USERNAME=mehulbafnaa
# {"username":"mehulbafnaa","key":"a36eb14443aa54eaf661c9833140edd0"}
export KAGGLE_KEY=a36eb14443aa54eaf661c9833140edd0

curl -L -u $KAGGLE_USERNAME:$KAGGLE_KEY\
  -o model.tar.gz\
  https://www.kaggle.com/api/v1/models/google/gemma-3/flax/gemma3-4b/1/download