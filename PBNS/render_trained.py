import argparse
import os
import glob
import numpy as np
import sys
from Data.smpl.smpl_np import SMPLModel
from Model.PBNS import PBNS

from IO import readOBJ, writeOBJ
from util import loadInfo, quads2tris, weights_prior
from values import rest_pose


def with_ones(X):
    ones = np.ones((*X.shape[:2], 1), dtype=np.float32)
    return np.concatenate((X, ones), axis=-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render poses from a dataset.")
    parser.add_argument("--npz_dump_dir", type=str, default="./output/npz_dump")
    parser.add_argument("--body_mat", type=str, default="./Model/Body.mat")
    parser.add_argument("--output_dir", type=str, default="./output/trained")
    parser.add_argument("--smpl_path", type=str, default="./Data/smpl/model_f.pkl")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Read body data
    model_smpl = SMPLModel(model_path=args.smpl_path, rest_pose=rest_pose)

    # Read outfit data
    object = "Outfit"
    body = os.path.basename(args.body_mat).split('.')[0]
    blendweights = False
    model_outfit = PBNS(object=object, body=body, checkpoint=None, blendweights=False)
    body_shape = model_outfit._shape

    outfit_F = model_outfit._F
    body_F = model_outfit._body_faces

    npz_paths = sorted(glob.glob(os.path.join(args.npz_dump_dir, "*.npz")))
    print("Found {} npz files in {}".format(len(npz_paths), args.npz_dump_dir))
    assert len(npz_paths) > 0, f"No npz files found in {args.npz_dump_dir}"

    n_every_frame = 10
    for npz_path in npz_paths:
        print("Rendering {}".format(npz_path))
        bname = os.path.basename(npz_path).split(".npz")[0]
        data = np.load(npz_path)
        poses = data["poses"]
        G = data["G"]
        body = data["body"]
        preds = data["pred"] # outfit

        # print("  - poses: {}".format(poses.shape))
        # print("  - G: {}".format(G.shape))
        # print("  - body: {}".format(body.shape))
        # print("  - pred: {}".format(preds.shape))
        for batch_idx in range(poses.shape[0]):
            if batch_idx % 8 != 0:
                continue

            pose = poses[batch_idx]
            pred = preds[batch_idx]

            # save body
            G, B = model_smpl.set_params(pose=pose, beta=body_shape, with_body=True)
            save_path = os.path.join(args.output_dir, f"{bname}_{batch_idx:02d}_body.obj")
            writeOBJ(save_path, B, body_F)

            # save outfit
            G, B = model_smpl.set_params(pose=pose, beta=body_shape, with_body=True)
            save_path = os.path.join(args.output_dir, f"{bname}_{batch_idx:02d}_outfit.obj")
            writeOBJ(save_path, pred, outfit_F)
            
            print("Saved body and outfit for pose {}: {}".format(batch_idx, bname))
            
    print("DONE")