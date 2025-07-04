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
    v_template = body_data.get('v_template', None)
    return body_T, faces, body_W, shape, v_template

def read_outfit(outfit_path):
    outfit_T, outfit_F_tri = readOBJ(outfit_path)
    outfit_F = quads2tris(outfit_F_tri) # triangulate
    outfit_W = weights_prior(outfit_T, body_T, body_W)
    outfit_W /= np.sum(outfit_W, axis=-1, keepdims=True)

    return outfit_T, outfit_F, outfit_W

def with_ones(X):
    ones = np.ones((*X.shape[:2], 1), dtype=np.float32)
    return np.concatenate((X, ones), axis=-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render poses from a dataset.")
    parser.add_argument("--npz_dir", type=str, default="./output/npz_dump")
    parser.add_argument("--body_mat", type=str, default="./Model/Body.mat")
    parser.add_argument("--outfit_path", type=str, default="./Model/Outfit.obj")
    parser.add_argument("--output_dir", type=str, default="./output/debug/train")
    parser.add_argument("--smpl_path", type=str, default="./Data/smpl/model_f.pkl")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Read body data
    body_T, body_F, body_W, body_shape, v_template = read_body(args.body_mat)

    # Load SMPL model
    SMPL = SMPLModel(model_path=args.smpl_path, rest_pose=body_T)

    # Read outfit data
    outfit_T, outfit_F, outfit_W = read_outfit(args.outfit_path)
    outfit_T = outfit_T[None, :, :]

    # Save rest pose
    save_path = os.path.join(args.output_dir, '{}.obj'.format(os.path.basename(args.body_mat).split('.')[0]))
    writeOBJ(save_path, body_T, body_F)
    print(f"Saved body mesh to {save_path}")

    # npz_paths
    npz_paths = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    npz_paths = [npz_paths[-1]]
    print(f"Found {len(npz_paths)} files in {args.npz_dir}")
    print()
    # body_W shape: TensorShape([12037, 24])
    # G shape: TensorShape([16, 24, 4, 4])
    # T shape: TensorShape([16, 12037, 3])
    for i, npz_path in enumerate(npz_paths):
        print("[{}/{}]: {}".format(i + 1, len(npz_paths), npz_path))
        data = np.load(npz_path)

        poses = data['poses']
        G = data['G']
        body = data['body']
        pred = data['pred']
        
        print("G:", G.shape)
        print("body:", body.shape)
        print("pred:", pred.shape)
        print("poses:", poses.shape)
        # skinning
        G = np.einsum('ab,cbde->cade', outfit_W, G)
        outfit_skinned = np.einsum('abcd,abd->abc', G, with_ones(outfit_T))[:, :, :3]

        for batch_idx in range(outfit_skinned.shape[0]):
            body_name = f"{os.path.basename(npz_path).split('.')[0]}_{batch_idx:02d}_body.obj"
            outfit_name = f"{os.path.basename(npz_path).split('.')[0]}_{batch_idx:02d}_outfit.obj"
            save_path_body = os.path.join(args.output_dir, body_name)
            save_path_outfit = os.path.join(args.output_dir, outfit_name)

            """ CREATE BODY AND OUTFIT .OBJ FILES """
            writeOBJ(save_path_body, v_template, body_F)
            writeOBJ(save_path_outfit, outfit_skinned, outfit_F)

            print("  -", save_path)

    print("DONE")