# Put embedding conversions in this file, and definitions
import groove.downbeats
import numpy as np
from typing import Callable
import scipy, math
from scipy.signal import butter, lfilter


def butter_lopass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_hipass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='high', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lopass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_hipass_filter(data, cutoff, fs, order=5):
    b, a = butter_hipass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
  
def butter_bandpass_filter(data, locut, hicut, fs, order=5):
  return butter_lowpass_filter(
          butter_hipass_filter(data, hicut, fs, order),
                                     locut, fs, order)

# data is raw data
# dbeats is something like bn[[bn[:,1] == 1, 0] where bn is output of beatnet
# bar_num is the measure to process
# dimension is the number of divisions of the bar
def bar_embedding(data,dbeats,bar_num,dimension,framerate,kernel=None, kernel_width=None, square=True):
    assert bar_num < len(dbeats), 'bar_num must be smaller than the number of bars in the audio'

    time_interval = (dbeats[bar_num-1],dbeats[bar_num])
    frame_interval = (int(time_interval[0]*framerate), int(time_interval[1]*framerate))

    sub_beats = np.round(np.linspace(frame_interval[0],frame_interval[1],dimension+1))
    sub_beat_interval = int(sub_beats[1] - sub_beats[0])

    if not kernel:
        if not kernel_width:
            kernel_width = 1

        assert kernel_width > 0
        if kernel_width < 1:
            kernel_sigma = kernel_width * sub_beat_interval
        else:
            kernel_sigma = framerate * kernel_width / 1000
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
        if square:
            sub_data = sub_data**2
        # print(np.sum(sub_data),np.sum(kernel))
        sub_beat_data[i] = np.sum(kernel*(sub_data))

    return sub_beat_data 

# data is raw data
# dbeats is something like bn[[bn[:,1] == 1, 0] where bn is output of beatnet
# bar_num is the measure to process
# dimension is the number of divisions of the bar
def bar_embedding_freq(samples,proc, dbeats,bar_num,subdivisions,
                  framerate: float,kernel=None, kernel_width=None, square=True):
    """_summary_

    Args:
        samples (np.array): All audio samples in track
        dbeats (_type_): List of locations of downbeats
        bar_num (_type_): _description_
        dimension (_type_): Number of divisions per bar
        framerate (float): Sample rate per second
        kernel (_type_, optional): _description_. Defaults to None.
        kernel_width (_type_, optional): _description_. Defaults to None.
        square (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    assert bar_num < len(dbeats), 'bar_num must be smaller than the number of bars in the audio'

    time_interval = (dbeats[bar_num-1],dbeats[bar_num])
    frame_interval = (int(time_interval[0]*framerate), int(time_interval[1]*framerate))

    sub_beats = np.round(np.linspace(frame_interval[0],frame_interval[1],subdivisions+1))
    sub_beat_interval = int(sub_beats[1] - sub_beats[0])

    if not kernel:
        if not kernel_width:
            kernel_width = 1
        kernel_sigma = kernel_width*sub_beat_interval
        kernel = np.exp(-np.arange(-sub_beat_interval,sub_beat_interval,1)**2/(2*kernel_sigma**2))
        kernel = kernel / np.sum(kernel)
    # print(kernel.shape)


    # power = groove.downbeats.smooth_power(samples, framerate)
    
    
    sub_beat_data = [0]*(subdivisions * 3) # we do not want to count down beat twice
    for i in range(subdivisions):
        # getting data around subbeat[i] of length 2*sub_beat_length
        # print(sub_beats[i])
        start = int(sub_beats[i]-sub_beat_interval)
        end = int(sub_beats[i]+sub_beat_interval)

        powers_segs = list(map(lambda pow: pad_with_zeros(pow, start, end, sub_beat_interval), proc))
        
        for j in range(len(powers_segs)):
            sub_beat_data[i + j] = np.sum(kernel*(powers_segs[j]))

    return sub_beat_data 

def pad_with_zeros(samples, start, end, sub_beat_interval):
    sub_data = np.zeros(2*sub_beat_interval)
    if start < 0:
        sub_data[-start:] = samples[0:end]
    elif end > len(samples):
        sub_data[:len(samples)-end] = samples[start:]
    else:
        sub_data = samples[start:end]
    return sub_data

def get_freq_segmented_powers(samples, framerate, locut=200, midrange=[400,5000], hicut=5000):
    lo_samples = butter_lowpass_filter(samples, locut, framerate)
    mid_samples = butter_bandpass_filter(samples, *midrange, framerate)
    hi_samples = butter_hipass_filter(samples, hicut, framerate)
    return list(map(lambda s: groove.downbeats.smooth_power(s, framerate), 
                                        [lo_samples, mid_samples, hi_samples]))

def load_bar_embedding(file, divisions, weights, process: Callable, ext="mp3", square=True):
    beat_data = groove.downbeats.get_beat_data(file)
    _, proc, sr = groove.downbeats.get_audio_data(file, process, ext=ext)

    db = beat_data[beat_data[:,1] == 1, 0]
    sub_beat_data = []
    for bar_num in range(1,db.shape[0]):
        p = []
        for i, division in enumerate(divisions):
            p.append(np.array(bar_embedding(proc/max(abs(proc)), db, bar_num=bar_num, dimension=division, framerate=sr, kernel_width=1/4, square=square)) * weights[i])
        sub_beat_data.append(np.concatenate(p, axis=0))

    return np.stack(sub_beat_data, axis=0)

def load_bar_embedding_freq(file, divisions, weights, 
                            process: Callable = get_freq_segmented_powers, 
                            kernel_width=30, ext="mp3", square=True):
    beat_data = groove.downbeats.get_beat_data(file)
    rawdata, proc, sr = groove.downbeats.get_audio_data(file, process, ext=ext)

    db = beat_data[beat_data[:,1] == 1, 0]
    sub_beat_data = []
    for bar_num in range(1,db.shape[0]):
        p = []
        for i, division in enumerate(divisions):
            p.append(np.array(bar_embedding_freq(rawdata, proc, db, bar_num=bar_num, 
                                subdivisions=division, framerate=sr, 
                                kernel_width=kernel_width, square=square)) * weights[i])
        sub_beat_data.append(np.concatenate(p, axis=0))

    return np.stack(sub_beat_data, axis=0)

# Resamples bars so they all have the same number of samples
# Number of samples is the maximum number which is a multiple of divisor
# Returns resampled data and number of samples per bar
def uniformize_bars(data, dbeats, sr, divisor = 1):
    n_bars = dbeats.shape[0]
    if n_bars < 2:
        min_bar_samples = data.shape[0]
    else:
        min_bar_samples = int((dbeats[1:] - dbeats[:-1]).min().item() * sr)
    # Force an extra multiple of two because we subdivide by half
    assert divisor < min_bar_samples, f'lcm {divisor} must be smaller than smallest bar length {min_bar_samples}'
    bar_samples = (min_bar_samples // divisor) * divisor

    resampled_data = np.zeros(0)
    # Resample each bar to be uniform
    for i in range(dbeats.shape[0]):
        start = int(dbeats[i] * sr)
        if i < dbeats.shape[0] - 1:
            end = int(dbeats[i+1] * sr)
        else:
            end = data.shape[0]
        resampled_data = np.concatenate([resampled_data, scipy.signal.resample(data[start:end], bar_samples)], axis=0)

    return resampled_data, bar_samples


# divisions is list of bar subdivisions
# If kernel width is in (0, 1) then interpret it as a fraction of interval
# If kernel width is >= 1 then interpret it as a time in milliseconds
def bar_embedding_total(data, dbeats, divisions, sr, kernel=None, kernel_width=1/4):
    
    # Figure out the number of samples in a standard rescaled bar; make it a multiple of lcm of dimension
    n_bars = dbeats.shape[0]
    if n_bars == 0:
        print("NO BARS??!??!")
    if n_bars == 1:
        print("ONLY ONE BAR???")

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

    #resampled_data, new_samples = uniformize_bars(data, dbeats, sr, dim_lcm)

    # Kernels for each division
    lens = []
    kernels = []
    for d in divisions:
        sub_beat_interval = new_samples // (d * 2)
        lens.append(sub_beat_interval)

        assert kernel_width > 0
        if kernel_width < 1:
            kernel_sigma = kernel_width * sub_beat_interval
        else:
            kernel_sigma = sr * kernel_width / 1000

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


def load_bar_embedding_total(file, divisions, weights, process: Callable, kernel_width=1/4, ext="mp3", concatenate=True):
    beat_data = groove.downbeats.get_beat_data(file)
    _, proc, sr = groove.downbeats.get_audio_data(file, process, ext=ext)

    db = beat_data[beat_data[:,1] == 1, 0]

    embeds = bar_embedding_total(proc/max(abs(proc)), db, divisions, sr, kernel_width=1/4)
    for i, e in enumerate(embeds):
        embeds[i] = embeds[i] * weights[i]

    if concatenate:
        return np.concatenate(embeds, axis=1)

    
    return embeds
