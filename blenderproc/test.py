import h5py
import numpy as np
import json

with h5py.File("output/0.hdf5") as f:
    colors = np.array(f["colors"])
    print(colors.shape)
    print(f)
    text = np.array(f["object_states"]).tostring()
    obj_states = json.loads(text)