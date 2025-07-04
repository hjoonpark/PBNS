import argparse
import os
import glob
import numpy as np
import sys
from Data.smpl.smpl_np import SMPLModel

from IO import readOBJ, writeOBJ
from util import loadInfo, quads2tris, weights_prior

def read_body(body_mat):
    if not body_mat.endswith('.mat'): body_mat = body_mat + '.mat'
    body_data = loadInfo(os.path.abspath(os.path.dirname(__file__)) + '/' + body_mat)
    for k, v in body_data.items():
        if isinstance(v, np.ndarray):
            print('  - {} : {}'.format(k, v.shape))
        else:
            print('  - {} : {}'.format(k, v))

    body_W = body_data.get('blendweights', None)
    faces = body_data.get('faces', None)
    shape = body_data.get('shape', None)
    body_T = body_data.get('body', None)
    return body_T, faces, body_W, shape

def with_ones(X):
    ones = np.ones((*X.shape[:2], 1), dtype=np.float32)
    return np.concatenate((X, ones), axis=-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render poses from a dataset.")
    parser.add_argument("--pose_rootdir", type=str, default="../data_CMU/CMU")
    parser.add_argument("--pose_subdir", type=str, default="01")
    parser.add_argument("--body_mat", type=str, default="./Model/Body.mat")
    parser.add_argument("--output_dir", type=str, default="./output/poses")
    parser.add_argument("--smpl_path", type=str, default="./Data/smpl/model_f.pkl")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pose_dir = os.path.join(args.pose_rootdir, args.pose_subdir)
    pose_paths = sorted(glob.glob(os.path.join(pose_dir, "*.npz")))
    print(f"Found {len(pose_paths)} pose files in {pose_dir}")
    
    body_mat = args.body_mat
    
    # Read body data
    body_T, body_F, body_W, body_shape = read_body(body_mat)

    SMPL = SMPLModel(model_path=args.smpl_path, rest_pose=body_T)

    n_every_frame = 10
    for root, _, files in os.walk(args.pose_rootdir):
        for file in files:
            if file.endswith(".npz"):
                path = os.path.join(root, file)
                data = np.load(path)
                if 'poses' not in data:
                    print(f"[Skipping] No 'poses' in {path}")
                    continue
                poses = data['poses'][:, :72]
                n_poses = poses.shape[0]
                if n_poses < n_every_frame:
                    continue

                for pose_idx, pose in enumerate(poses):
                    if pose_idx % n_every_frame != 0:
                        continue

                    # save body
                    # skinning
                    G, B = SMPL.set_params(pose=pose, beta=body_shape, with_body=True)
                    save_path = os.path.join(args.output_dir, f"{os.path.basename(file).split('.')[0]}_{pose_idx}_body.obj")
                    writeOBJ(save_path, B, body_F)
                    print(f'  pose {pose_idx}: {save_path}')
                print("Exit")
                exit(0)
                


    print("DONE")