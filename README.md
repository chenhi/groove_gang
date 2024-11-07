# groove_gang

## Folders:
* Root folder is a sandbox/testing ground, use it for messing around
* groove/ contains "library" files (import them)
* data/ contains various processed data

## Code: 
* groove/downbeats.py contains functions used to extract the downbeats from audio data, downbeats_demo.ipynb is a demo
* groove/intraclustering.py contains clustering functions used to extract "primary" beats from songs (needs a good embedding for good results)
* groove/embedding.py contains implementation of some embeddings

## Data:
* data/beatnet_data.pkl is a dictionary containing BeatNet outputs for some songs
* data/groovemidi_data.pkl is a dictionary containing BeatNet outputs for the Groove MIDI dataset: https://magenta.tensorflow.org/datasets/groove


