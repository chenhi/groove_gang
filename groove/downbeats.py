import pickle as pkl
import librosa
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from typing import Callable



# Loads file audio and BeatNet data, slices into measures, processes and then returns
def get_measures(file: str, process: Callable, ext="mp3"):

    # Get BeatNet output
    with open("beatnet_data.pkl","rb") as f:
        data = pkl.load(f)
    beat_data = data[file]

    # Get raw audio data
    y, sr = librosa.load(f'beatnet/inputs/{file}.{ext}')
    # Apply chosen processing
    y_proc = process(y, sr)

    # Cut into measures
    downbeats = beat_data[beat_data[:,1] == 1, 0]
    downbeat_frames = (downbeats * sr).astype(int)
    raw_measures = []
    proc_measures = []
    for i in range(0, downbeat_frames.shape[0] - 1):
        raw_measures.append(y[downbeat_frames[i]:downbeat_frames[i+1]])
        proc_measures.append(y_proc[downbeat_frames[i]:downbeat_frames[i+1]])

    return raw_measures, proc_measures, sr



# Gets the beat times for the processed measures using a beat finding function
def get_beat_times(proc_measures, sr, beat_find: Callable, ext="mp3"):
    beat_times = []
    for i in range(len(proc_measures)):
        beat_times.append(beat_find(proc_measures[i], sr))

    return beat_times






# Processing functions
def smooth_power(y, sr):
    power_db = y**2
    return savgol_filter(power_db, sr*0.01, delta=1/sr, polyorder=2, deriv=0,mode='constant')

def log_smooth_power(y, sr):
    power_db = y**2
    return np.log10(.0001 + savgol_filter(power_db, sr*0.01, delta=1/sr, polyorder=2, deriv=0,mode='constant'))





# Beat finding functions
def max_pool(x, k):
    x = np.append(x, np.zeros(x.shape[0]//k * k + k - x.shape[0]))
    return x.reshape(-1, k).max(axis=1)

def beat_stdev(m, sr):
    bf = 1. * (m > np.mean(m) + 2 * np.std(m))
    k = int(sr * 0.05)
    bf = max_pool(bf, k)
    # Get the frame indices (rescale up)
    return np.arange(bf.shape[0])[bf == 1] * k

def beat_peaks(m, sr):
    return find_peaks(m, height=np.mean(m) + 2 * np.std(m), distance=int(sr*0.05), prominence=np.std(m))[0]





