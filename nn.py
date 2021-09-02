import torch

class LossTracker:
	def __init__(self, lower_bound=None, cvg_slope=None, patience=None):
		self.bound = lower_bound
		self.slope = cvg_slope
		self.patience = patience
		self.reset()

	def reset(self):
		self.__epoch = 0
		self._losses = []
		self._slope_iter = 0

	def reset_slope_iter(self):
		self._slope_iter = 0

	def append(self, loss):
		'''
			Sums the given loss to a accumulator. Used in the minibatch loop.
		'''
		self.__epoch += loss.item()

	def step(self, dataset_size):
		'''
			Averages the accumulated loss with given dataset size and evaluates the relative slope. Used in the epoch loop, after iterating minibatches.
		'''
		l = self.__epoch / dataset_size
		self._losses.append(l)
		self.__epoch = 0
		if len(self._losses) > 1 and (self._losses[-2]-self._losses[-1])/self._losses[-1] < self.slope:  # Relative difference (slope).
			self._slope_iter += 1
		else:
			self._slope_iter = 0
		return l

	@property
	def losses(self):
		'''
			Recorded losses list.
		'''
		return self._losses

	def is_bound(self):
		'''
			Returns True if lower bound is set and the last recorded loss was below or equal to it.
		'''
		return self.bound is not None and self._losses[-1] <= self.bound

	def is_converged(self):
		'''
			Returns True if the last recorded relative slope was under the bound for beyond patience number of times.
		'''
		return self._slope_iter > self.patience
