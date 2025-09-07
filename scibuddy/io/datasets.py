import torch
import numpy as np

from .loaders import load_split, cameraposes_loader, pointcloud_loader, images_loader, collate_index_pose_image, multi_loader


class MVRSilhouetteDataset(torch.utils.data.Dataset):
	'''
		Multi view reconstruction from silhouettes.
	'''
	def __init__(self, split_file, data_dir='data', force_shape=None):
		self.silh_files = load_split(split_file, prefix=data_dir)
		self.force_shape = force_shape

	def __len__(self):
		return len(self.silh_files)

	def __getitem__(self, i):
		'''
			Reads all images and camera poses for given model as a batch.
		'''
		f = self.silh_files[i]
		poses = torch.from_numpy(cameraposes_loader(f).astype(np.float32))
		images = torch.stack(images_loader(f, filename='silhouette*', force_shape=self.force_shape))
		return i, poses, images

	def collate(self, samples):
		return collate_index_pose_image(samples)

class MVRColorDataset(torch.utils.data.Dataset):
	'''
		Multi view reconstruction from RGB images.
	'''
	def __init__(self, split_file, data_dir='data', force_shape=None):
		self.files = load_split(split_file, prefix=data_dir)
		self.force_shape = force_shape

	def __len__(self):
		return len(self.files)

	def __getitem__(self, i):
		'''
			Reads all images and camera poses for given model as a batch.
		'''
		f = self.files[i]
		poses = torch.from_numpy(cameraposes_loader(f).astype(np.float32))
		images = torch.stack(images_loader(f, filename='rgb*', force_shape=self.force_shape))
		return i, poses, images

	def collate(self, samples):
		return collate_index_pose_image(samples)

class PointCloudDataset(torch.utils.data.Dataset):
	def __init__(self, split_file, data_dir='data'):
		self.files = load_split(split_file, prefix=data_dir)

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		f = self.files[idx]
		return torch.from_numpy(pointcloud_loader(f).vertices).float()

	def collate(self, samples):
		return torch.vstack(samples)

class OrientedPointCloudDataset(torch.utils.data.Dataset):
	def __init__(self, split_file, data_dir='data'):
		self.files = load_split(split_file, prefix=data_dir)

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		f = self.files[idx]
		return torch.from_numpy(pointcloud_loader(f).vertices).float(), torch.from_numpy(pointcloud_loader(f, 'normalcloud').vertices).float()

	def collate(self, samples):
		points = []
		normals = []
		for s in samples:
			pc, nrm = s
			points.append(pc)
			normals.append(nrm)
		return torch.vstack(points), torch.vstack(normals)

class GenericDataset(torch.utils.data.Dataset):
	def __init__(self, dataset_dir, split_file, loaders):
		with open(split_file, 'rb') as f:
			files = []
			for l in f:
				files.append(l.strip())
		self.classes, self.models, self.data = multi_loader(dataset_dir, files, loaders)

	def __len__(self):
		return len(self.classes)

	def __get__(self, i):
		return self.classes[i], self.models[i], self.data[i]

class MergeDataset(torch.utils.data.Dataset):
	def __init__(self, datasets):
		self.datasets = datasets

		l = [len(d) for d in datasets]
		if min(l) != max(l):
			raise RuntimeError('Datasets have different lengths!')

	def __len__(self):
		return len(self.datasets[0])

	def __getitem__(self, i):
		pieces = [d[i] for d in self.datasets]
		return pieces

	def collate(self, samples):
		data = []
		for i in range(len(self.datasets)):
			d = []
			for s in samples:
				d.append(s[i])
			data.append(self.datasets[i].collate(d))
		return data
