# groove_gang

## Folders:
* Root folder is a sandbox/testing ground, use it for messing around
* groove/ contains "library" files (import them)
* beatnet/ contains instructions for installing BeatNet and some Jupyter notebooks for performing extraction.  This is isolated from the rest of the code because people have had difficulty installing BeatNet and we don't want to make it a prequisite for now
* beatnet/inputs/ contains audio files for processing.  It should be empty.  Don't push these!  Share it using other channels.

## Code: 
* groove/downbeats.py contains functions used to extract the downbeats from audio data, downbeats_demo.ipynb is a demo
* groove/intraclustering.ipynb contains clustering functions used to extract "primary" beats from songs (needs a good embedding for good results)
* Files in root folder beginning with demo_ are demos of the above

## Data:
* beatnet_data.pkl is a dictionary containing BeatNet outputs for some songs
* groovemidi_data.pkl is a dictionary containing BeatNet outputs for the Groove MIDI dataset: https://magenta.tensorflow.org/datasets/groove


