import h5py

with h5py.File('validate_record-three-armed-2019_11_29.hdf5', 'r') as file:
    key = file.keys()
    print()