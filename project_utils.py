import os
import re
import sys
import math
import json
import glob
import shutil

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import yaml

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
			pilimg = transforms.ToPILImage()(img.detach().cpu())
			pilimg.save(os.path.join(self.folder, 'img_{:05d}.png'.format(self.index)))
		elif isinstance(img, np.ndarray):
			pilimg = Image.fromarray(img)
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


def load_configfile(path):
	with open(path, 'r') as f:
		if path.endswith('.json'):
			return json.loads(re.sub("//.*", "", f.read(), flags=re.MULTILINE))  # Simulating comments.
		elif path.endswith('.yaml'):
			return yaml.load(f, yaml.FullLoader)
		else:
			raise RuntimeError('Unknown config file type: ' + str(path))

def get_git_commit_hash() -> str:
	return os.popen('git rev-parse HEAD').read().strip()

class Experiment:
	def __init__(self, configfile=None, param_index=1, name=None, root=None, group=None, version=None, timestamp=None, store_config=True, parameters_version=None):
		'''
			Create an experiment insance and load/set the configuration dictionary. Creates the experiment directory. Config file must contain a 

			Parameters
			----------
			configfile : {str, dict}
				Filepath of the JSON or YAML config file or the actual parameters dictionary (for runtime configuration). If None, attempt reading filepath from program parameters. (default: None)
			param_index : int
				Index of the config filepath in program parameters. (default: 1)
			name : str
				Name of the experiment, used as the prefix of the experiment. If None, attempt reading from loaded config. (default: None)
			root : str
				Path to the root folder that will contain experiments. If None, attempt reading from loaded config. (default: None)
			group : bool, optional
				Group project versions by project name (adds a directory level). If None, attempt reading from loaded config. (default: False)
			version : str, optional
				Append a version string to experiment directory name. If None, attempt reading from loaded config. If False, disable reading from config (use for sub-experiments). (default: None)
			timestamp : bool, optional
				Add a timestamp to project folder name (rudimentary experiment versioning, protects from overwritting results). If None, attempt reading from loaded config. (default: True)
			store_config: bool, optional
				Save the config file into the experiment folder. (default: True)
			parameters_version: str, optional
				Store the parameters version into the config to allow downstream versioning.
		'''
		if configfile is None:  # Attempt extracting from program arguments.
			if len(sys.argv) < param_index+1:
				print('Missing experiment parameters file!')
				exit(1)
			configfile = sys.argv[param_index]

		if isinstance(configfile, str):
			self.config = load_configfile(configfile)
		elif isinstance(configfile, dict):
			self.config = configfile
		else:
			raise RuntimeError('Unrecognized config file type: ' + str(type(configfile)))

		if group is None:
			group = self.config['experiment'].get('group', False)
		if version is None:
			version = self.config['experiment'].get('version', None)
		elif version is False:
			version = None
		if timestamp is None:
			timestamp = self.config['experiment'].get('timestamp', True)
		if name is None:
			name = self.config['experiment']['name']
		if root is None:
			root = self.config['experiment'].get('root', 'out')
		if parameters_version is not None:
			self.config['experiment']['parameters_version'] = parameters_version

		# Create experiment output folder.
		self.name = self.dir = name
		if group:  # Group dirs by name.
			self.dir = os.path.join(self.name, self.name)
		if version is not None:  # Append version to dir name.
			self.dir = self.dir + '_' + str(version)
		if timestamp:  # Append timestamp to dir name.
			from datetime import datetime
			self.dir = self.dir + '_' + datetime.now().strftime('%Y%m%d%H%M%S%f')
		self.dir = os.path.join(root, self.dir.replace(' ', '_'))
		os.makedirs(self.dir, exist_ok=True)

		if store_config:
			self.store_config(self.config)

	def store_config(self, config=None, use_yaml=True):
		"""
			Store experiment config dict.

			Parameters
			----------
			config: dict, optional
				Configuration dict to explicitly store (if different than experiment's). If None, store experiment's config. Default: None
		"""
		if config is None:
			config = self.config
		with open(self('parameters.yaml'), 'w') as f:
			if use_yaml:
				yaml.dump(config, f)
			else:
				json.dump(config, f, indent='\t')

	def __getitem__(self, k):
		return self.config[k]

	def __contains__(self, k):
		return k in self.config

	def __call__(self, *args):
		return self.path(*args)

	def __str__(self):
		'''
			Basename of the experiment directory path.
		'''
		return os.path.basename(self.dir)

	def new_subexperiment(self, config, name, group=True, version=None, timestamp=True, store_config=True):
		'''
			Create a sub-experiment within this one. Used for iterated development over components or over entire projects.

			For component-wise iteration, use parameters `group=True` and `timestamp=True`.
			This is by default.

			For project-wise iteration, use parameters `group=False` and `timestamp=False`.

		'''
		return Experiment(config, name=name, root=self.dir, group=group, version=version, timestamp=timestamp, store_config=store_config)

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

	def makedirs(self, dir):
		'''
			Creates the directories for given leaf directory path.

			Parameters
			----------

			dir : str
				Path to target file.

			Returns
			----------
			dir : str
				File path concatenated to experiment root.
		'''
		path = self.path(dir)
		os.makedirs(path, exist_ok=True)
		return path

	def path(self, *args):
		'''
			Join given path elements to the experiment directory path.
		'''
		return os.path.join(self.dir, *args)


class GridSearch:
	def __init__(self, parameters, grid):
		'''
			Iterator through all combinations of given parameter alternative values.

			Parameters
			----------
			parameters: {str, dict}
				A dictionary containing all parameters with default values. If string, it is treated as a filepath of the JSON or YAML config file.
			grid: {str, dict}
				A dictionary containing lists of possible values for each of the given parameters. If string, it is treated as a filepath of the JSON or YAML config file.
		'''
		if isinstance(parameters, str):
			self.parameters = load_configfile(parameters)
		elif isinstance(parameters, dict):
			self.parameters = parameters
		else:
			raise RuntimeError('Unrecognized grid type: ' + str(type(grid)))

		if isinstance(grid, str):
			self.grid = load_configfile(grid)
		elif isinstance(grid, dict):
			self.grid = grid
		else:
			raise RuntimeError('Unrecognized grid type: ' + str(type(grid)))

		self.__lengths = {k: len(self.grid[k]) for k in self.grid.keys()}

	def __len__(self):
		return math.prod(self.__lengths.values())

	def __iter__(self):
		self.__i = -1
		self.__idx = {k: 0 for k in self.grid.keys()}
		self.__idx[list(self.grid.keys())[0]] = -1  # Initial condition.
		return self

	def __next__(self):
		# Update indices.
		t = True
		for k in self.__idx.keys():
			self.__idx[k] += 1
			if self.__idx[k] == self.__lengths[k]:
				self.__idx[k] = 0
				continue
			t = False
			break
		if t:
			raise StopIteration

		# Update current parameter values.
		for k in self.grid.keys():
			v = self.grid[k][self.__idx[k]]
			if isinstance(v, dict):  # For parameter groups (parameters that must be used jointly).
				for kk, vv in v.items():
					self.parameters[kk] = vv
			else:  # For standalone parameters.
				self.parameters[k] = self.grid[k][self.__idx[k]]

		self.__i += 1
		return self.parameters, self.__i