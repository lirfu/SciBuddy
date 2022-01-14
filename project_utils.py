import os
import re
import sys
import json
import glob
import shutil

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

from .data_utils import orient_img
from .logging_utils import LOG

class CheckpointSaver:
	def __init__(self, ex, name=None):
		self.__min_loss = 'inf'
		if name is None:
			self.__filepath = ex.path('best_model.pt')
		else:
			self.__filepath = ex.path(f'best_model_{name}.pt')

	def save(self, dictionary, loss):
		if float(self.__min_loss) >= float(loss):
			self.__min_loss = loss
			torch.save(dictionary, self.__filepath)

	def load(self):
		if os.path.exists(self.__filepath):
			return torch.load(self.__filepath)
		return None

	@property
	def loss(self):
		return self.__min_loss


class GifMaker:
	def __init__(self, ex, name: str):
		self.index = 1
		self.folder = ex.makedir(ex.path(name))
		self.name = name
		self.ex = ex
		self.clear()

	def __call__(self, img):
		if isinstance(img, torch.Tensor):
			pilimg = transforms.ToPILImage()(orient_img(img.detach().cpu(), out=True))
			pilimg.save(os.path.join(self.folder, 'img_{:05d}.png'.format(self.index)))
		elif isinstance(img, np.ndarray):
			pilimg = Image.fromarray(orient_img(img, out=True))
			pilimg.save(os.path.join(self.folder, 'img_{:05d}.png'.format(self.index)))
		elif isinstance(img, str) and img == 'pyplot':
			plt.savefig(os.path.join(self.folder, 'img_{:05d}.png'.format(self.index)), bbox_inches='tight', pad_inches=0, transparent=False)
		else:
			raise RuntimeError('Unknown image type for GIF! ({})'.format(type(img)))

		self.index += 1

	def generate(self, duration_sec: int=10, loop: int=0):
		'''
			Read images from temporary folder and generate a GIF.
		'''
		l = [Image.open(f).convert('P') for f in sorted(glob.glob(os.path.join(self.folder,'img_*.png')))]
		if len(l) == 0:
			LOG.e(f'No images found to generate {self.name}.gif!')
			return
		img, *imgs = l
		if isinstance(duration_sec, int) or isinstance(duration_sec, float):  # Calculate global per-frame duration.
			duration = int(duration_sec*1000/(self.index-1))
		else:  # Convert per-frame to millicsconds.
			duration = [int(d*1000) for d in duration_sec]
		img.save(fp=self.ex.path(self.name+'.gif'), format='GIF', append_images=imgs, save_all=True, duration=duration, loop=loop, palette='RGB', interlace=False, optimize=False, include_color_table=True)

	def clear(self):
		self.index = 1
		shutil.rmtree(self.folder, ignore_errors=True)
		os.makedirs(self.folder)


class Experiment:
	def __init__(self, configfile=None, param_index=1, group=None, version=None, timestamp=None):
		'''
			Create an experiment insance and load/set the configuration dictionary. Creates the experiment directory. Config file must contain a 

			Parameters
			----------
			configfile : {str, dict}
				Filepath of the JSON config file or the actual config dictionary (for runtime configuration). If None, attempt reading filepath from program parameters. (default: None)
			param_index : int
				Index of the config filepath in program parameters. (default: 1)
			group : bool, optional
				Group project versions by project name (adds a directory level). If None, attempt reading from loaded config. (default: False)
			version : str, optional
				Append a version string to experiment directory name. If None, attempt reading from loaded config. (default: '')
			timestamp : bool, optional
				Add a timestamp to project folder name (rudimentary experiment versioning, protects from overwritting results). If None, attempt reading from loaded config. (default: True)
		'''
		if configfile is None:  # Attempt extracting from program arguments.
			if len(sys.argv) < param_index+1:
				print('Missing experiment parameters file!')
				exit(1)
			configfile = sys.argv[param_index]

		if isinstance(configfile, str):
			with open(configfile, 'r') as f:
				self.config = json.loads(re.sub("//.*", "", f.read(), flags=re.MULTILINE))
		elif isinstance(configfile, dict):
			self.config = configfile

		if group is None:
			group = self.config['experiment'].get('group', False)
		if version is None:
			version = self.config['experiment'].get('version', '')
		if timestamp is None:
			timestamp = self.config['experiment'].get('timestamp', True)

		# Create experiment output folder.
		self.name = self.dir = self.config['experiment']['name']
		if group:  # Group dirs by name.
			self.dir = os.path.join(self.name, self.name)
		if version is not None:  # Append version to dir name.
			self.dir = self.dir + '_' + str(version)
		if timestamp:  # Append timestamp to dir name.
			from datetime import datetime
			self.dir = self.dir + '_' + datetime.now().strftime('%Y%m%d%H%M%S')
		self.dir = os.path.join(self.config['experiment'].get('root', 'out'), self.dir.replace(' ', '_'))
		self.makedir(self.dir)

		# Store experiment params.
		with open(self.path('parameters.json'), 'w') as f:
			json.dump(self.config, f, indent='\t')

	def __getitem__(self, k):
		return self.config[k]

	def __contains__(self, k):
		return k in self.config

	def __str__(self):
		'''
			Basename of the experiment directory path.
		'''
		return os.path.basename(self.dir)

	def list_similar(self):
		'''
			Returns a list of experiment from parent directory, starting with the same experiment name.
		'''
		parent = os.path.dirname(self.dir)
		dirs = os.listdir(parent)
		return list(filter(lambda d: d.startswith(self.name), dirs))

	def get_last_similar(self):
		'''
			Returns the experiment path withing same parent directory, starting with the same experiment name, but latest version/timestamp.
		'''
		dirs = self.list_similar()
		sorted(dirs)
		return os.path.join(os.path.dirname(self.dir),dirs[-1])

	def pretty_config(self):
		strings = ['Configuration:']
		def r(d, ss, i=1):
			offset = '  '*i
			for k,v in d.items():
				if isinstance(v, dict):
					ss.append(f'{offset}- {k}:')
					r(v, ss, i+1)
				else:
					ss.append(f'{offset}- {k}: {v}')
		r(self.config, strings)
		return '\n'.join(strings)

	def exists(self, filename):
		return os.path.exists(self.path(filename))

	def makedir(self, dirname):
		'''
			Creates given directory path with missing parents. If directory exist, returns.

			Parameters
			----------

			dirname : str
				Path to target directory.

			Returns
			----------
			dirname : str
				Unmodified input, used for method chaining.
		'''
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		return dirname

	def path(self, *args):
		'''
			Join given path elements to the experiment directory path.
		'''
		return os.path.join(self.dir, *args)

	def save_image(self, img: torch.Tensor, filepath: str):
		'''
			Save given image to given filepath. Filepath gets appended to experiment path.
		'''
		if len(img.shape) == 2:
			img = img.reshape(*img.shape,1)
		img = transforms.ToPILImage()(orient_img(img, out=True))
		img.save(self.path(filepath), optimize=False, compress_level=0)

	def save_pyplot_image(self, name, transparent=False, **kwargs):
		'''
			Save pyplot buffer image to given filepath. Filepath gets appended to experiment path.
		'''
		plt.savefig(self.path(name), bbox_inches='tight', pad_inches=0, transparent=transparent, **kwargs)