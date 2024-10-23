# Put embedding conversions in this file, and definitions


import numpy as np

# Beat times:
# An array of variable size that lists the beat times

# Beat frames:
# An array of variable size that lists the various beat frames

# Toy embedding:
# For a given sample rate sr, and a time length t in seconds, is a shape (sr * t, ) tensor with a 1 if a beat is detected and 0 if not.



def beat_frames_to_toy(frames, len, target_len):
    x = np.zeros(len + ((target_len - len) % target_len))
    x[frames] = 1
    x = x.reshape((target_len, -1))
    x = x.max(axis=1)
    return x

