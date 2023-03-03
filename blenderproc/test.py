import h5py
import numpy as np
import json

with h5py.File("blenderproc/data/0.hdf5") as f:
    colors = np.array(f["colors"])
    text = np.array(f["ground_truth"]).tostring()
ground_truth = json.loads(text)
print(ground_truth)
