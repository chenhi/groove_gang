# Put embedding conversions in this file, and definitions
import groove.downbeats
import numpy as np
from typing import Callable
import scipy, math

# data is raw data
# dbeats is something like bn[[bn[:,1] == 1, 0] where bn is output of beatnet
# bar_num is the measure to process
# dimension is the number of divisions of the bar
def bar_embedding(data,dbeats,bar_num,dimension,framerate,kernel=None, kernel_width=None):
    assert bar_num < len(dbeats), 'bar_num must be smaller than the number of bars in the audio'

    time_interval = (dbeats[bar_num-1],dbeats[bar_num])
    frame_interval = (int(time_interval[0]*framerate), int(time_interval[1]*framerate))

    sub_beats = np.round(np.linspace(frame_interval[0],frame_interval[1],dimension+1))
    sub_beat_interval = int(sub_beats[1] - sub_beats[0])

    if not kernel:
        if not kernel_width:
            kernel_width = 1
        kernel_sigma = kernel_width*sub_beat_interval
        kernel = np.exp(-np.arange(-sub_beat_interval,sub_beat_interval,1)**2/(2*kernel_sigma**2))
        kernel = kernel / np.sum(kernel)
    # print(kernel.shape)

    
    sub_beat_data = [0]*(dimension) # we do not want to count down beat twice
    for i in range(dimension):
        # getting data around subbeat[i] of length 2*sub_beat_length
        # print(sub_beats[i])
        start = int(sub_beats[i]-sub_beat_interval)
        end = int(sub_beats[i]+sub_beat_interval)
        sub_data = np.zeros(2*sub_beat_interval)
        # print(sub_data.shape)
        # print(start,end)
        if start < 0:
            sub_data[-start:] = data[0:end]
        elif end > len(data):
            sub_data[:len(data)-end] = data[start:]
        else:
            sub_data = data[start:end]

        #print(data.shape, sub_data.shape)

        sub_data = sub_data**2
        # print(np.sum(sub_data),np.sum(kernel))
        sub_beat_data[i] = np.sum(kernel*(sub_data))

    return sub_beat_data 


def bar_embedding_total(data, dbeats, divisions, sr, kernel=None, kernel_width=1/4):
    
    # Figure out the number of samples in a standard rescaled bar; make it a multiple of lcm of dimension
    n_bars = dbeats.shape[0]
    min_bar_samples = int((dbeats[1:] - dbeats[:-1]).min().item() * sr)
    # Force an extra multiple of two because we subdivide by half
    dim_lcm = math.lcm(*list(divisions)) * 2
    assert dim_lcm < min_bar_samples, f'lcm {dim_lcm} must be smaller than smallest bar length {min_bar_samples}'
    new_samples = (min_bar_samples // dim_lcm) * dim_lcm
    
    resampled_data = np.zeros(0)
    # Resample each bar to be uniform
    for i in range(dbeats.shape[0]):
        start = int(dbeats[i] * sr)
        if i < dbeats.shape[0] - 1:
            end = int(dbeats[i+1] * sr)
        else:
            end = data.shape[0]
        resampled_data = np.concatenate([resampled_data, scipy.signal.resample(data[start:end], new_samples)], axis=0)


    # Kernels for each division
    lens = []
    kernels = []
    for d in divisions:
        sub_beat_interval = new_samples // (d * 2)
        lens.append(sub_beat_interval)

        kernel_sigma = kernel_width * sub_beat_interval
        kernel = np.exp(-np.arange(-sub_beat_interval,sub_beat_interval,1)**2/(2*kernel_sigma**2))
        kernels.append(kernel / np.sum(kernel))

    outputs = [np.zeros(0) for _ in divisions]
    # Handle the first bar (edge case, half-cut off)
    for i, d in enumerate(divisions):
        outputs[i] = np.append(outputs[i], (kernels[i][-lens[i]:] * resampled_data[0:lens[i]]).sum())
        
    # Reshape the rest and apply kernel
    for i, d in enumerate(divisions):
        divided_data = resampled_data[lens[i]:-lens[i]].reshape(-1, 2 * lens[i])
        outputs[i] = np.append(outputs[i], (divided_data * kernels[i]).sum(axis=1))
        # Reshape the outputs to group by bar
        outputs[i] = outputs[i].reshape((-1, d))
        
    return outputs


def load_bar_embedding(file, divisions, weights, process: Callable, ext="mp3"):
    beat_data = groove.downbeats.get_beat_data(file)
    _, proc, sr = groove.downbeats.get_audio_data(file, process, ext=ext)

    db = beat_data[beat_data[:,1] == 1, 0]
    sub_beat_data = []
    for bar_num in range(1,db.shape[0]):
        p = []
        for i, division in enumerate(divisions):
            p.append(np.array(bar_embedding(proc/max(abs(proc)),db,bar_num=bar_num,dimension=division,framerate=sr,kernel_width=1/4)) * weights[i])
        sub_beat_data.append(np.concatenate(p, axis=0))

    return np.stack(sub_beat_data, axis=0)

def load_bar_embedding_total(file, divisions, weights, process: Callable, ext="mp3", concatenate=True):
    beat_data = groove.downbeats.get_beat_data(file)
    _, proc, sr = groove.downbeats.get_audio_data(file, process, ext=ext)

    db = beat_data[beat_data[:,1] == 1, 0]

    embeds = bar_embedding_total(proc/max(abs(proc)), db, divisions, sr)
    for i, e in enumerate(embeds):
        embeds[i] = embeds[i] * weights[i]

    if concatenate:
        return np.concatenate(embeds, axis=1)
        
    
    return embeds
