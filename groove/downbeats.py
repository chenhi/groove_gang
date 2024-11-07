import pickle as pkl
import librosa
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from typing import Callable


def get_audio_data(file: str, process: Callable, ext="mp3"):
    y, sr = librosa.load(f'inputs/{file}.{ext}')
    # Apply chosen processing
    return y, process(y, sr), sr


def get_beat_data(file: str):
    # Get BeatNet output
    with open("data/beatnet_data.pkl","rb") as f:
        data = pkl.load(f)
    return data[file]


# Loads file audio and BeatNet data, slices into measures, processes and then returns
def get_measures(file: str, process: Callable, ext="mp3"):

    # Get BeatNet output
    beat_data = get_beat_data(file)

    # Get raw audio data
    y, y_proc, sr = get_audio_data(file, process, ext)

    # Cut into measures
    downbeats = beat_data[beat_data[:,1] == 1, 0]
    downbeat_frames = (downbeats * sr).astype(int)
    raw_measures = []
    proc_measures = []
    for i in range(0, downbeat_frames.shape[0] - 1):
        raw_measures.append(y[downbeat_frames[i]:downbeat_frames[i+1]])
        proc_measures.append(y_proc[downbeat_frames[i]:downbeat_frames[i+1]])

    return raw_measures, proc_measures, sr


# Gets the beat times frames and total frames for the processed measures using a beat finding function
def get_beat_frames(proc_measures: list[np.ndarray], sr: int, beat_find: Callable):
    beat_frames = []
    num_frames = []
    for i in range(len(proc_measures)):
        beat_frames.append(beat_find(proc_measures[i], sr))
        num_frames.append(proc_measures[i].shape[0])

    return beat_frames, num_frames







# Processing functions
def smooth_power(y, sr):
    power_db = y**2
    return savgol_filter(power_db, sr*0.01, delta=1/sr, polyorder=2, deriv=0, mode='constant')

def log_smooth_power(y, sr):
    power_db = y**2
    return np.log10(.0001 + savgol_filter(power_db, sr*0.01, delta=1/sr, polyorder=2, deriv=0,mode='constant'))

def smooth_power2(y, sr):
    power_db = y**2
    d2 = savgol_filter(power_db, sr*0.01, delta=1/sr, polyorder=3, deriv=2, mode='constant') ** 2
    return d2/d2.max()




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





