import os
import numpy as np

data_dir = "data_CMU/CMU"
pose_list = []

for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".npz"):
            path = os.path.join(root, file)
            data = np.load(path)
            if 'poses' not in data:
                print(f"[Skipping] No 'poses' in {path}")
                continue
            print(f"Processing: {path}")
            poses = data['poses'][:, :72]
            pose_list.append(poses)

# Save result
save_dir = "Data"
os.makedirs(save_dir, exist_ok=True)
train_data = np.concatenate(pose_list, axis=0)
np.save(f"{save_dir}/train.npy", train_data)
print("Saved train.npy with shape:", train_data.shape)
