import random
import numpy as np
import torch


def test_print(a, name='array'):
	'''
		Prints stuff about the given torch.Tensor: name, value range, is any NaN, are all NaN, shape.
		Returns the same array for convenience.
	'''
	print('{}   [{:+.3e}, {:+.3e}]   ({},{})   {}'.format(name, a.min().item(), a.max().item(), a.isnan().any().item(), a.isnan().all().item(), a.shape))
	return a
	

def reproducibility(seed):
	'''
		Sets random value generator states of Python.random, numpy.random and torch.random to given seed value.
	'''
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	#torch.set_deterministic(True)
	torch.backends.cudnn.deterministic = True


class ReproducibleContext:
	def __init__(self, seed, device=None):
		'''
			Create a reproducible context that guarantees consistent random value generator state of Python.random, numpy.random and torch.random.
			On exit, restores the previous state of generators.

			Parameters
			----------
			seed : int or None
				If integer, is used as seed for the random generators. If None, behaves as if it doesn't exist (convenience behaviour for random experiments).
			device : str
				Torch device of interest. If starts with 'cuda', random state of that CUDA device will be altered. Default: None
		'''
		self.seed = seed
		self.device = device

	def __enter__(self):
		if self.seed is not None:
			self.rd_state = random.getstate()
			self.np_state = np.random.get_state()
			self.tr_state = torch.get_rng_state()
			if self.device is not None and self.device[:4] == 'cuda':
				self.cuda_state = torch.cuda.get_rng_state(self.device)

			reproducibility(self.seed)

	def __exit__(self, type, value, trace):
		if self.seed is not None:
			random.setstate(self.rd_state)
			np.random.set_state(self.np_state)
			torch.set_rng_state(self.tr_state)
			if self.device is not None and self.device[:4] == 'cuda':
				torch.cuda.set_rng_state(self.cuda_state, self.device)


def get_device(use_gpu=True):
	'''
		Returns the torch device, preferring CUDA if required.
	'''
	if use_gpu and torch.cuda.is_available():
		return torch.device('cuda')
	else:
		return torch.device('cpu')
