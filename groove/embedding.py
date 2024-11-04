# Put embedding conversions in this file, and definitions


import numpy as np

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