import os
import re
import copy
import math
import json
import glob
import shutil
from typing import Union, Sequence, Any, List

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import yaml

from .logging import LOG


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

	def generate(self, duration_sec:Union[int,Sequence[int]]=10, loop:int=0):
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
		"""
		l = [Image.open(f).convert('RGB', dither=Image.Dither.NONE) for f in sorted(glob.glob(os.path.join(self.directory,'img_*.png')))]
		if len(l) == 0:
			LOG.e(f'No images found to generate {self.name}.gif!')
			return
		img, *imgs = l
		if isinstance(duration_sec, int) or isinstance(duration_sec, float):  # Calculate global per-frame duration.
			duration = int(duration_sec*1000/(self.index-1))
		else:  # Convert per-frame to millicsconds.
			duration = [int(d*1000) for d in duration_sec]
			print(self.name, imgs, )
		img.save(fp=self.ex.path(self.name+'.gif'), format='GIF', append_images=imgs, save_all=True, duration=duration, loop=loop, interlace=False, optimize=False)

	def clear(self):
		self.index = 1
		shutil.rmtree(self.directory, ignore_errors=True)
		os.makedirs(self.directory)


def load_configfile(path:str):
	"""
		Load the configuration dictionary from filepath.
	"""
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
	def __init__(
		self,
		configfile:Union[str,dict],
		name:str=None,
		root:str=None,
		group:bool=None,
		version:str=None,
		timestamp:bool=None,
		store_config:bool=True,
		parameters_version:str=None,
		reuse:str=None
	):
		"""
			Create an experiment insance and prepare&save the configuration dictionary.
			Creates the experiment directory.
			Parameter values from constructor are preferred to ones from config, unless they werent specified.

			Parameters
			----------
			configfile : {str, dict}
				Filepath of the JSON or YAML config file or the actual parameters dictionary.
				The `experiment` section is reserved for experiment tracking and some keys might be overwritten by this initialization (see code).
			name : str
				Name of the experiment, used as the prefix of the experiment.
				If None, attempt reading from loaded config.
				Default: None
			root : str
				Path to the root directory that will contain experiments.
				If None, attempt reading from loaded config.
				Default: None
			group : bool, optional
				Group experiment versions by experiment name (adds a directory level).
				If None, attempt reading from loaded config.
				Default: False
			version : str, optional
				Append a version string to experiment directory name.
				If None, attempt reading from loaded config.
				If False, disable reading from config (use for sub-experiments).
				Default: None
			timestamp : bool, optional
				Add a timestamp to experiment directory name (rudimentary experiment versioning, protects from overwritting results).
				If None, attempt reading from loaded config.
				Default: True
			store_config: bool, optional
				Save the config file into the experiment directory.
				Default: True
			parameters_version: str, optional
				Store the parameters version into the config to allow downstream versioning.
				Default: None
			reuse : str
				Path to existing experiment directory, e.g. for continuing the experiment series if interrupted.
				If None, a new directory will be generated from given parameters.
				Default: None
		"""
		# Resolve the config parameters.
		if isinstance(configfile, str):
			self.config = load_configfile(configfile)
		elif isinstance(configfile, dict):
			self.config = configfile
		else:
			raise RuntimeError('Unrecognized config file type: ' + str(type(configfile)))

		# Resolve experiment name parameters.
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

		# Prepare experiment directory.
		if reuse is None:  # Create the new experiment output directory.
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
		else:  # Use the provided experiment directory.
			self.name = name
			self.dir = reuse

	def store_config(self, config=None, use_yaml=True):
		"""
			Store experiment config dict into `root/parameters.yaml` (or `.json`).

			Parameters
			----------
			config: dict, optional
				Configuration dict to explicitly store (if different than experiment's). If None, store experiment's config. Default: None
			use_yaml: bool, optional
				Store as YAML. Otherwise, stored as json. Default: True
		"""
		if config is None:
			config = self.config
			if use_yaml:
				with open(self('parameters.yaml'), 'w') as f:
					yaml.dump(config, f, sort_keys=False)
			else:
				with open(self('parameters.json'), 'w') as f:
					json.dump(config, f, indent='\t', sort_keys=False)

	def __getitem__(self, k:str) -> Any:
		"""
			Gets the configuration value at given key.
		"""
		return self.config[k]

	def __contains__(self, k:str) -> bool:
		"""
			Checks if configuration file contains given key.
		"""
		return k in self.config

	def __call__(self, *args:Sequence[str]):
		"""
			Construct a path within experiment from given sequence of arguments.
		"""
		return self.path(*args)

	def __str__(self) -> str:
		"""
			Basename of the experiment directory path.
		"""
		return os.path.basename(self.dir)

	def new_subexperiment(self, config, name, group=True, version=None, timestamp=True, store_config=True) -> 'Experiment':
		"""
			Create a sub-experiment within this one. Used for iterated development over components or over entire experiments.

			For component-wise iteration, use parameters `group=True` and `timestamp=True`.
			This is by default.

			For experiment-wise iteration, use parameters `group=False` and `timestamp=False`.

		"""
		return Experiment(config, name=name, root=self.dir, group=group, version=version, timestamp=timestamp, store_config=store_config)

	def list_similar(self) -> Sequence[str]:
		"""
			Returns a list of experiment paths from parent directory, starting with the same experiment name.
		"""
		parent = os.path.dirname(self.dir)
		dirs = os.listdir(parent)
		return list(filter(lambda d: d.startswith(self.name), dirs))

	def get_last_similar(self) -> str:
		"""
			Returns the experiment path within same parent directory, starting with the same experiment name, but latest version/timestamp.
		"""
		dirs = self.list_similar()
		sorted(dirs)
		return os.path.join(os.path.dirname(self.dir),dirs[-1])

	def pretty_config(self) -> str:
		"""
			Constructs a pretty string of the configuraiton dictionary.
		"""
		strings = ['Configuration:']
		def r(d, ss, i=1):
			offset = '  '*i
			for k,v in d.items():
				if isinstance(v, dict):
					ss.append(f'{offset}{k}:')
					r(v, ss, i+1)
				else:
					ss.append(f'{offset}{k}: {v}')
		r(self.config, strings)
		return '\n'.join(strings)

	def exists(self, filename:str) -> bool:
		"""
			Checks if given filename exists within the experiment directory.
		"""
		return os.path.exists(self.path(filename))

	def makedirs(self, dir:str) -> str:
		"""
			Creates the directories for given leaf directory path.
			Appends directories to the experiment directory.

			Parameters
			----------

			dir : str
				Path to target directory.

			Returns
			----------
			dir : str
				File path appended to experiment root.
		"""
		path = self.path(dir)
		os.makedirs(path, exist_ok=True)
		return path

	def prepare_path(self, *path) -> str:
		"""
			Creates the predating directories supporting the given leaf file (create all of the directories except the last file/directory).

			Parameters
			----------

			path : args[str]
				Path to target file.

			Returns
			----------
			path : str
				Original path appended to experiment root.
		"""
		path = self.path(os.path.join(*path))
		dir = os.path.dirname(path)
		os.makedirs(dir, exist_ok=True)
		return path

	def path(self, *args) -> str:
		"""
			Construct a path within experiment from given sequence of arguments.
		"""
		return os.path.join(self.dir, *args)


class GridSearch:
	def __init__(self, parameters:dict, grid:dict, skip_indices:Union[int,List[int]]=0):
		"""
			Iterator through all combinations of given parameter alternative values.
			The parameters are set in order of their definition.

			Parameter groups can be defined by wrapping the grouped parameters into a dict, the parameters in a group will all be set in a parameter instance.
			Previously set parameters are keeped between groups (if a group sets a unique parameter, it is persisted throughout the rest of the groups).

			Parameters
			----------
			parameters: {str, dict}
				A dictionary containing all parameters with default values. If string, it is treated as a filepath of the JSON or YAML config file.
			grid: {str, dict}
				A dictionary containing lists of possible values for each of the given parameters. If string, it is treated as a filepath of the JSON or YAML config file.
			skip_indices: Union[int,List[int]]
				When integer, number of iterations to skip (used when continuing work when one experiment died and stopped the loop). When list of integers, skips experiments with given indices (counting from 1). If 0, runs the entire grid. Default: 0
		"""
		if isinstance(parameters, str):
			self.parameters = load_configfile(parameters)
		elif isinstance(parameters, dict):
			self.parameters = parameters
		else:
			raise RuntimeError('Unrecognized parameters type: ' + str(type(parameters)))

		if isinstance(grid, str):
			self.grid = load_configfile(grid)
		elif isinstance(grid, dict):
			self.grid = grid
		else:
			raise RuntimeError('Unrecognized grid type: ' + str(type(grid)))

		self.__lengths = {k: len(self.grid[k]) for k in self.grid.keys()}
		if isinstance(skip_indices, int):
			self.__skip_indices = list(range(skip_indices + 1))
		else:
			self.__skip_indices = skip_indices

	def __len__(self):
		return math.prod(self.__lengths.values())

	def __iter__(self):
		self.i = 0
		self.__idx = {k: 0 for k in self.grid.keys()}
		self.__idx[list(self.grid.keys())[0]] = -1  # Initial condition.
		return self

	def __next__(self):
		while True:
			self.i += 1
			# Update grid indices.
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
			if self.i not in self.__skip_indices:
				break

		# Update current parameter values.
		parameters = copy.deepcopy(self.parameters)  # For parameter groups that don't define a parameter, the original is used.
		changes = {}
		for k in self.grid.keys():
			v = self.grid[k][self.__idx[k]]
			if isinstance(v, dict):  # For parameter groups (parameters that must be used jointly).
				for kk, vv in v.items():
					parameters[kk] = vv
					changes[kk] = vv
			else:  # For standalone parameters.
				parameters[k] = v
				changes[k] = v

		return parameters, changes
