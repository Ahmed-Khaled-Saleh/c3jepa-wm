import h5py, glob
import numpy as np
import os
def save_rollout_hdf5(rollout_idx, save_dict, data_dir):
    """Save one rollout as a group inside a shared HDF5 file."""
    os.makedirs(data_dir, exist_ok=True)
    h5_path = os.path.join(data_dir, "dataset.h5")

    with h5py.File(h5_path, 'a') as f:   # 'a' = append mode
        grp = f.create_group(f"rollout_{rollout_idx:06d}")
        for k, v in save_dict.items():
            grp.create_dataset(
                k,
                data=v if isinstance(v, np.ndarray) else np.asarray(v),
                compression="lz4",   # fast compression, good for uint8 images
            )


def merge_npz_to_hdf5(data_dir: str, out_path: str):
    """One-time merge of all .npz rollouts into a single indexed .h5 file."""

    files = sorted(glob.glob(os.path.join(data_dir, "rollout_*.npz")))
    with h5py.File(out_path, 'w') as h5:
        for fpath in files:
            name = os.path.splitext(os.path.basename(fpath))[0]
            data = np.load(fpath, allow_pickle=True)
            grp  = h5.create_group(name)
            for k in data.files:
                arr = data[k]
                
                if arr.ndim == 0:
                    # Save scalar metadata as an HDF5 attribute on the group instead of a dataset
                    grp.attrs[k] = arr.item()  # .item() converts numpy scalar to native Python type
                else:
                    # Save actual trajectory arrays as compressed datasets
                    # grp.create_dataset(k, data=arr, compression="lz4")
                    grp.create_dataset(k, data=arr, compression="gzip", compression_opts=4)

    print(f"Merged {len(files)} rollouts → {out_path}")

if __name__ == "__main__":
    merge_npz_to_hdf5(data_dir="./data/rollouts", out_path="./data/dataset.h5")