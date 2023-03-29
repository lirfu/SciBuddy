import glob
import os
import json

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import trimesh


def torch_float_to_torch_uint8(img):
	return (img * 255.).to(torch.uint8)

def chw_to_hwc(img: torch.Tensor):
	l = len(img.shape)
	if l == 2:
		return img.unsqueeze(2)
	elif l == 3:
		return img.moveaxis(0,2)
	elif l == 4:
		return img.moveaxis(1,3)
	else:
		raise RuntimeError('Unrecognized image shape: ' + str(l))


def hwc_to_chw(img: torch.Tensor):
	l = len(img.shape)
	if l == 2:
		return img.unsqueeze(0)
	elif l == 3:
		return img.moveaxis(2,0)
	elif l == 4:
		return img.moveaxis(3,1)
	else:
		raise RuntimeError('Unrecognized image shape: ' + str(l))

def rb_to_rgb(img):
	return torch.stack([img[...,0,:,:], torch.zeros(img.shape[-2:]), img[...,1,:,:]], dim=0)

def gray_to_rgb(img):
	l = len(img.shape)
	if l == 2:
		return img.unsqueeze(0).expand(3,-1,-1)
	elif l == 3:
		if img.shape[0] == 1:
			return img.expand(3,-1,-1)
	elif l == 4:
		if img.shape[1] == 1:
			return img.expand(-1,3,-1,-1)
	else:
		raise RuntimeError('Unrecognized image shape: ' + str(l))
	return img

def orient_img(img, to_chw=False, flip=False):
	if to_chw:
		if flip:
			img = img.flip((0))
		return img.permute(2,0,1)
	else:
		if flip:
			img = img.flip((1))
		return img.permute(1,2,0)

def load_image(filepath, shape=None, convert=None, flip=False, as_hwc=False, numpy=False):
	'''
		Loads image into a torch.Tensor.

		Parameters
		----------
		filepath: str
			Path to the source image.

		shape: Tuple(2), optional
			Target HW shape of the image after loading. If `None`, original size is kept. Default: None

		convert: str, optional
			Target Pillow format to which loaded image is converted.
			Common values: `L` for 8-bit grayscale, `RGB` for 8-bit color, `I` for 32-bit integers,
			`F` for 32-bit floats. Check Pillow documentation on 'Modes' for more options. If `None`,
			original (Pillow automatic) format is kept. Default: None

		flip: bool, optional
			Flips image vertically to match the right coordinate system. Default: False

		as_hwc: bool, optional
			Swaps the channels dimension to the end. Default: False

		numpy: bool, optional
			Returns the image as a numpy.ndarray. Otherwise, as a torch.Tensor. Default: False
	'''
	if not os.path.exists(filepath):
		raise RuntimeError('Image file not found:', filepath)
	img = Image.open(filepath)
	if img.mode == 'P':  # Pallete mode should be converted to RGBA, otherwise defaults to grayscale.
		img = img.convert('RGBA')
	if convert is not None:
		img = img.convert(convert)
	if shape is not None:
		img = img.resize(shape[::-1])
	if numpy:
		img = np.array(img)  # HWC
		if flip:
			img = np.flip(img, 0)
		if not as_hwc and len(img) == 3:
			img = np.moveaxis(img, 2, 0)
		return img
	else:
		img = transforms.ToTensor()(img)  # CHW, [0,1]
		if flip:
			img = img.flip((1))
		if as_hwc:
			return chw_to_hwc(img)
	return img

def save_image(filepath, img, **kwargs):
	directory = os.path.dirname(filepath)
	if directory != '':
		os.makedirs(directory, exist_ok=True)
	if isinstance(img, np.ndarray):
		img = Image.fromarray(img)
	elif isinstance(img, torch.Tensor):
		img = transforms.ToPILImage()(img)
	else:
		raise RuntimeError('Unknown image type', type(img))
	img.save(filepath, optimize=kwargs.get('optimize', False), compress_level=kwargs.get('compress_level', 0))

class ImageCat:
	"""
		Utility for concatenating images along a dimension.
	"""
	def __init__(self):
		self._images = []

	def __call__(self, img: torch.Tensor) -> None:
		self._images.append(img)

	def generate(self, dim: int=2) -> torch.Tensor:
		if len(self._images) == 0:
			return torch.tensor([[0]])
		return torch.cat(self._images, dim=dim)

class ImageGridCat:
	"""
		Utility for concatenating images in a grid of specified shape. Expects images of shape (C,H,W)
	"""
	def __init__(self, shape):
		self.h_cat = ImageCat()
		self.v_cat = ImageCat()
		self.__index = 0
		self.shape = shape

	def __call__(self, img: torch.Tensor) -> None:
		self.h_cat(img)
		self.__index += 1
		if self.__index == self.shape[1]:
			self.__index = 0
			self.v_cat(self.h_cat.generate(2))
			self.h_cat = ImageCat()

	def generate(self) -> torch.Tensor:
		return self.v_cat.generate(1)


### DATA LOADING ###


def find_files(d, name, extensions):
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

