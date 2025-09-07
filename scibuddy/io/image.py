import os

import numpy as np
from PIL import Image
import torch
import torchvision as tv


def torch_float_to_torch_uint8(img):
	"""
		Coverts floating point from range [0,1] to uint8 in range [0,255].
	"""
	return (img * 255.99).to(torch.uint8)

def chw_to_hwc(img: torch.Tensor):
	"""
		Converts the image dimension ordering, handling both individual and batched images.
	"""
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
	"""
		Converts the image dimension ordering, handling both individual and batched images.
	"""
	l = len(img.shape)
	if l == 2:
		return img.unsqueeze(0)
	elif l == 3:
		return img.moveaxis(2,0)
	elif l == 4:
		return img.moveaxis(3,1)
	else:
		raise RuntimeError('Unrecognized image shape: ' + str(l))

def gray_to_rgb(img):
	"""
		Converts grayscale image to RGB by repeating the same channel 3 times.
	"""
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
		img = tv.transforms.ToTensor()(img)  # CHW, [0,1]
		if flip:
			img = img.flip((1))
		if as_hwc:
			return chw_to_hwc(img)
	return img

def save_image(filepath, img, **kwargs):
	"""
		Saves image given as numpy array (HWC) or torch tensor (CHW) to the given file path.
	"""
	directory = os.path.dirname(filepath)
	if directory != '':
		os.makedirs(directory, exist_ok=True)
	if isinstance(img, np.ndarray):
		img = Image.fromarray(img)
	elif isinstance(img, torch.Tensor):
		img = tv.transforms.ToPILImage()(img)
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
