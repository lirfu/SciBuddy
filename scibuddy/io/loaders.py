import glob
import os
import json

import numpy as np
import torch
import trimesh

from .image import load_image


def find_files(d, name, extensions):
	"""
		Finds all files in given directory having the given name and any of given extensions.
	"""
	if isinstance(extensions, str):
		extensions = [extensions]
	files = []
	for e in extensions:
		files.extend(glob.glob(os.path.join(d, name+'.'+e)))
	return files

def mesh_loader(model_dir, filename='model', force_mesh=True, extensions=['obj','ply']) -> trimesh.Trimesh:
	files = find_files(model_dir, filename, extensions)

	if len(files) > 1:
		print(f'Found {len(files)} matching files! Choosing file {files[0]} from {files}.')
	if len(files) == 0:
		raise RuntimeError(f'Found no files matching: {filename}.{extensions}')

	force = 'mesh' if force_mesh else None
	model = trimesh.load(files[0], force=force)  # TODO Here losing texturing on force. Try with PyTorch3D!

	if len(model.faces) == 0:
		raise RuntimeError(f'Model contains 0 faces (probably loaded a point cloud): {files[0]}')
	return model

def pointcloud_loader(model_dir, filename='pointcloud', extensions=['ply','npy']) -> trimesh.PointCloud:
	files = find_files(model_dir, filename, extensions)

	if len(files) > 1:
		print(f'Found {len(files)} matching files! Choosing file {files[0]} from {files}.')
	if len(files) == 0:
		raise RuntimeError(f'Found no files matching: {filename}.{extensions}')

	f = files[0]
	if f.endswith('.ply'):
		pc = trimesh.load(f)
	else:
		pc = trimesh.PointCloud(np.load(f))

	if hasattr(pc, 'faces'):
		raise RuntimeError(f'Model contains faces (probably loaded a mesh): {f}')
	if len(pc.vertices) == 0:
		raise RuntimeError(f'Model contains 0 vertices (unknown reason): {f}')
	return pc

def voxel_loader(model_dir, filename='voxel', extensions=['binvox']) -> trimesh.PointCloud:
	files = find_files(model_dir, filename, extensions)

	if len(files) > 1:
		print(f'Found {len(files)} matching files! Choosing file {files[0]} from {files}.')
	if len(files) == 0:
		raise RuntimeError(f'Found no files matching: {filename}.{extensions}')

	return trimesh.exchange.binvox.load_binvox(files[0])

def images_loader(model_dir, filename='image_rgba.*', force_shape=None, extensions=['png','jpg']) -> torch.Tensor:
	files = find_files(model_dir, filename, extensions)

	if len(files) == 0:
		raise RuntimeError(f'Found no files matching: {filename}.{extensions}')

	sorted(files)  # Sort by name to ensure consistency.
	images = []
	for f in files:
		images.append( load_image(f, shape=force_shape) )

	return images

def cameraposes_loader(model_dir, filename='camera_poses', extensions='npy') -> np.ndarray:
	files = find_files(model_dir, filename, extensions)

	if len(files) == 0:
		raise RuntimeError(f'Found no files matching: {filename}.{extensions}')

	if len(files) > 1:
		print(f'Found {len(files)} matching files! Choosing file {files[0]} from {files}.')
	return np.load(files[0])


def multi_loader(dataset_dir, files, loader_fns):
	if not os.path.exists(dataset_dir):
		raise RuntimeError('Path does not exist:', dataset_dir)

	if callable(loader_fns):
		loader_fns = [loader_fns]

	if len(loader_fns) == 0:
		raise RuntimeError('Specify loader functions!')

	c = []
	m = []
	data = []
	for f in files:
		pth = os.path.join(dataset_dir,f)
		dat = []
		for l in loader_fns:
			d = l(pth)
			if d:
				dat.append(d)
		c.append(dataset_dir)
		m.append(f)
		data.append(dat)
	return c, m, data


class TransformZXYtoXYZ:
	def __init__(self, device):
		self.matrix = torch.tensor([
			[ 1,  0,  0],
			[ 0,  0,  1],
			[ 0, -1,  0]
		]).float().to(device)

	def __call__(self, pts):
		return torch.matmul(pts, self.matrix)


def load_split(split_file, prefix=None):
	files = []
	with open(split_file, 'r') as f:
		if split_file.endswith('.lst'):
			for l in f:
				if prefix:
					files.append(os.path.join(prefix, l.strip()))
				else:
					files.append(l.strip())
		elif split_file.endswith('.json'):
			d = json.load(f)
			for k,v in d.items():
				for model in v:
					if prefix:
						files.append(os.path.join(prefix,k,model))
					else:
						files.append(os.path.join(k,model))
		else:
			raise RuntimeError('Unknown split file format:', split_file)
	return files


def collate_index_pose_image(samples):
	'''
		Concatenates data batches of multiple models.
	'''
	indices = []
	poses = []
	images = []
	for i, po, im in samples:
		indices.append(i)
		poses.append(po)
		images.append(im)
	return torch.tensor(indices), torch.cat(poses, dim=0), torch.cat(images, dim=0)
