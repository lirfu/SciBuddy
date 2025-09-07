from typing import Union, Sequence

import os
import glob
import shutil

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

from ..tools.project import Experiment


class GifMaker:
	"""
		Generates a GIF from given sequence of images.
	"""
	def __init__(self, ex:'Experiment', name:str):
		self.index = 1
		self.directory = ex.makedirs(name)
		self.name = name
		self.ex = ex
		self.clear()

	def __call__(self, img:Union[torch.Tensor,np.ndarray,str]):
		"""
			Add image to GIF sequence.
			The image is stored as a file in the directory <gif_name>.
			If `img` is a torch tensor or numpy array, stores it as a PNG image,
			If `img` is a string 'pyplot', stores the current pyplot buffer as a PNG image.
		"""
		if isinstance(img, torch.Tensor):
			pilimg = transforms.ToPILImage()(img.detach().cpu())
			pilimg.save(os.path.join(self.directory, 'img_{:05d}.png'.format(self.index)))
		elif isinstance(img, np.ndarray):
			pilimg = Image.fromarray(img)
			pilimg.save(os.path.join(self.directory, 'img_{:05d}.png'.format(self.index)))
		elif isinstance(img, str) and img == 'pyplot':
			plt.savefig(os.path.join(self.directory, 'img_{:05d}.png'.format(self.index)), bbox_inches='tight', pad_inches=0, transparent=False)
		else:
			raise RuntimeError('Unknown image type for GIF! ({})'.format(type(img)))
		self.index += 1

	def generate(self, duration_sec:Union[int,Sequence[int]]=10, loop:int=0) -> bool:
		"""
			Read images from the <gif_name> directory and generate a GIF.

			Parameters
			----------
			duration_sec : int, list(int), optional
				Total GIF duration in seconds or the per-frame durations in seconds.
				Default: 10
			loop : int, optional
				Number of loops through all the entire GIF.
				If `0`, loops indefinitely.
				Default: 0

			Returns
			-------
			success : bool
                Returns `True` only if there were issues.
		"""
		l = [Image.open(f).convert('RGB', dither=Image.Dither.NONE) for f in sorted(glob.glob(os.path.join(self.directory,'img_*.png')))]
		if len(l) == 0:
			print(f'No images found to generate {self.name}.gif!')
			return True
		img, *imgs = l
		if isinstance(duration_sec, int) or isinstance(duration_sec, float):  # Calculate global per-frame duration.
			duration = int(duration_sec*1000/(self.index-1))
		else:  # Convert per-frame to millicsconds.
			duration = [int(d*1000) for d in duration_sec]
			print(self.name, imgs, )
		img.save(fp=self.ex.path(self.name+'.gif'), format='GIF', append_images=imgs, save_all=True, duration=duration, loop=loop, interlace=False, optimize=False)
		return False

	def clear(self):
		self.index = 1
		shutil.rmtree(self.directory, ignore_errors=True)
		os.makedirs(self.directory)
