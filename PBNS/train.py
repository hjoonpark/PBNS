import os
import sys
import numpy as np
import tensorflow as tf

from time import time
from datetime import timedelta
from math import floor

from Data.data import Data
from Model.PBNS import PBNS
from Losses import *
# use 'import' below for parallelized collision loss (specially for bigger batches)
# from LossesRay import *

from util import *
from IO import writePC2Frames

# ------------------------------------------------------------------
# tiny helper to save an OBJ (used only if you want the mesh dump)
def _save_obj(path, verts, faces):   # >>> ADDED
    with open(path, "w") as f:
        for v in verts: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces: f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
# ------------------------------------------------------------------

print("Available physical devices:", tf.config.list_physical_devices())

output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

""" PARSE ARGS """
gpu_id, name, object, body, checkpoint = parse_args()
if checkpoint is not None:
	checkpoint = os.path.abspath(os.path.dirname(__file__)) + '/checkpoints/' + checkpoint

ckpt_dir = os.path.join(output_dir, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)

# >>> ADDED: where to dump tensors & meshes
save_step = 5000          # save every N optimisation steps
npz_dir   = os.path.join(output_dir, "npz_dump")
os.makedirs(npz_dir, exist_ok=True)

""" GPU """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

""" Log """
stdout_steps = 100 # update stdout every N steps
if name == 'test': stdout_steps = 1

""" TRAIN PARAMS """
batch_size = 4
num_epochs = 10 if checkpoint is None else 10000

""" SIMULATION PARAMETERS """
edge_young = 15
bend_weight = 5 * 1e-5
collision_weight = 25
collision_dist = .004
mass = .3 # fabric mass as surface density (kg / m2)
blendweights = False # optimize blendweights?

""" MODEL """
print("Building model...")
model = PBNS(object=object, body=body, checkpoint=checkpoint, blendweights=blendweights)
tgts = model.gather() # model weights
tgts = [v for v in tgts if v.trainable]
model_summary(tgts)
optimizer = tf.optimizers.Adam()

""" DATA """
print("Reading data...")
tr_poses = 'Data/train.npy'
tr_data = Data(tr_poses, model._shape, model._gender, batch_size=batch_size)

tr_steps = floor(tr_data._n_samples / batch_size)
for epoch in range(num_epochs):
	print("")
	print("Epoch " + str(epoch + 1))
	print("--------------------------")
	""" TRAIN """
	print("Training...")
	total_time = 0
	step = 0
	metrics = [0] * 4 # Edge, Bend, Gravity, Collisions
	start = time()
	for poses, G, body in tr_data._iterator:
		""" Train step """
		with tf.GradientTape() as tape:
			pred = model(poses, G)

			# Losses & Metrics			
			# cloth
			L_edge, E_edge = edge_loss(pred, model._edges, model._E, weights=model._config['edge'])
			if bend_weight:
				L_bend, E_bend = bend_loss(pred, model._F, model._neigh_F, weights=model._config['bend'])
			else:
				L_bend, E_bend = 0, tf.constant(0)
			# collision
			L_collision, E_collision = collision_loss(pred, model._F, body, model._body_faces, model._config['layers'], thr=collision_dist)
			# gravity
			L_gravity, E_gravity = gravity_loss(pred, surface=model._total_area, m=mass)
			# pin
			if 'pin' in model._config:
				L_pin = tf.reduce_sum(tf.gather(model.D, model._config['pin'], axis=1) ** 2)
			else:
				L_pin = tf.constant(0.0)
			loss =  edge_young * L_edge + \
					bend_weight * L_bend + \
					collision_weight * L_collision + \
					L_gravity + \
					1e1 * L_pin
		""" Backprop """
		grads = tape.gradient(loss, tgts)		
		optimizer.apply_gradients(zip(grads, tgts))
		""" Progress """
		metrics[0] += E_edge.numpy()
		metrics[1] += E_bend.numpy()
		metrics[2] += E_gravity.numpy()
		metrics[3] += E_collision
		total_time = time() - start
		
		ETA = (tr_steps - step - 1) * (total_time / (1+step))
		if (step + 1) % stdout_steps == 0:
			sys.stdout.write('\r\tStep: ' + str(step+1) + '/' + str(tr_steps) + ' ... '
					+ 'E: {:.2f}'.format(1000 * metrics[0] / (1+step)) # in millimeters
					+ ' - '
					+ 'B: {:.3f}'.format(metrics[1] / (1+step))
					+ ' - '	
					+ 'G: {:.4f}'.format(1000 * metrics[2] / (1+step)) # in mJ
					+ ' - '
					+ 'C: [' + ', '.join(['{:.4f}'.format(m / (1+step)) for m in metrics[3]]) + ']'
					+ ' ... ETA: ' + str(timedelta(seconds=ETA)))
			sys.stdout.flush()


		if (step + 1) % 10000 == 0:
			model.save(os.path.join(ckpt_dir, name))
			print("  Checkpoint saved at step " + str(step + 1) + " to " + os.path.join(ckpt_dir, name))


		# ---------- inside training loop ---------------------------------
		if (step + 1) % save_step == 0:                      # >>> ADDED
			# 1. save weights
			model.save(os.path.join(ckpt_dir, name))
			print("\n  ✔︎ checkpoint saved at step", step+1)

			# 2. dump tensors to npz
			npz_path = os.path.join(npz_dir, f"{name}_ep{epoch+1:03d}_st{step+1:07d}.npz")
			np.savez_compressed(
				npz_path,
				poses=poses.numpy(),
				G=G.numpy(),
				body=body.numpy(),
				pred=pred.numpy()
			)
			print(f"  ✔︎ tensors saved → {os.path.basename(npz_path)}")

			# 3. quick OBJ of first predicted sample (optional)
			obj_path = os.path.join(
				npz_dir, f"{name}_ep{epoch+1:03d}_st{step+1:07d}.obj")
			_save_obj(obj_path, pred.numpy()[0], model._F)
			print(f"  ✔︎ mesh OBJ saved → {os.path.basename(obj_path)}")


		step += 1
	""" Epoch results """
	metrics = [m / step for m in metrics]
	print("")
	print("Total edge: {:.5f}".format(1000 * metrics[0]))
	print("Total bending: {:.5f}".format(metrics[1]))
	print("Total gravity: {:.2f}".format(metrics[2]))
	print("Total collision: [" + ', '.join(['{:.4f}'.format(m) for m in metrics[3]]) + ']')
	print("Total time: " + str(timedelta(seconds=total_time)))
	print("")
	""" Save checkpoint """
	model.save(os.path.join(ckpt_dir, name))