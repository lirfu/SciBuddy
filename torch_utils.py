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

def reproducibility(seed: int, device: str='cpu', force_determinism: bool=False):
	'''
		Sets random value generator states of Python.random, numpy.random and torch.random to given seed value.

		Parameters
		----------
		seed : int
			Seed value for the generators.
		device : str, optional
			Non-cpu device string to apply the seed to.
			Choose between: cpu, cuda, cuda:<device-id>.
			Default: cpu
		force_determinism : bool, optional
			Forces usage of deterministic implementations of algorithms when available.
				For non-deterministic algorithms issues a warning.
				Use only when required as it reduces overall performance.
				Default: False
	'''
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	device = str(device)
	if device.startswith('cuda'):
		if device == 'cuda':
			index = 0
		else:
			index = int(device.split(':')[1])
		torch.cuda.set_device(index)
		torch.cuda.manual_seed(seed)
	torch.use_deterministic_algorithms(force_determinism, warn_only=True)


class ReproducibleContext:
	def __init__(self, seed, device: str='cpu', index=0, force_determinism=False):
		'''
			Create a reproducible context that guarantees consistent random value generator state of Python.random, numpy.random and torch.random.
			On exit, restores the previous state of generators.

			Parameters
			----------
			seed : int or None
				If integer, is used as seed for the random generators.
				If None, behaves as if it doesn't exist (convenience behaviour for random experiments).
			device : str or None
				Torch device of interest.
				If 'cuda', additionally sets generator states of all CUDA devices.
				If only starts with 'cuda', additionally sets generator state of that specific CUDA device.
				Default: 'cpu'
			index : int
				Index of the device being used. Default: 0
			force_determinism : bool, optional
				Forces usage of deterministic implementations of algorithms when available.
				For non-deterministic algorithms issues a warning.
				Use only when required as it reduces overall performance.
				Default: False
		'''
		self.seed = seed
		self.device = str(torch.device(device, index))
		self.force_determinism = force_determinism

	def __enter__(self):
		if self.seed is not None:
			self.rd_state = random.getstate()
			self.np_state = np.random.get_state()
			self.tr_state = torch.get_rng_state()
			if self.device.startswith('cuda'):
				self.cuda_state = torch.cuda.get_rng_state(self.device)
			reproducibility(self.seed, self.device, self.force_determinism)

	def __exit__(self, type, value, trace):
		if self.seed is not None:
			random.setstate(self.rd_state)
			np.random.set_state(self.np_state)
			torch.set_rng_state(self.tr_state)
			if self.device.startswith('cuda'):
				torch.cuda.set_rng_state(self.cuda_state, self.device)


def get_device(device: str, index=0):
	'''
		Returns the torch device based on given string name and index (if applicable).
	'''
	if not isinstance(device, str):
		return device
	if device == 'cuda':
		return torch.device('cuda', index)
	else:
		return torch.device('cpu')
