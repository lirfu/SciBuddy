from typing import List

class LossTracker:
	def __init__(self, lower_bound: float=None, cvg_slope: float=1e-3, patience: int=1):
		'''
			Parameters
			----------
			lower_bound: float, optional
				Lower loss bound, used for detection of target loss. Not used if `None`. Default: `None`
			cvg_slope: float, optional
				Relative loss slope (absolute change divided by newer value), used for detection of convergence. Not used if `None`. Default: 1e-3
			patience: int, optional
				Number of iterations that forgive detections, used for ignoring noise in the loss curve. Not used if `None`. Default: 1
		'''
		self.bound = lower_bound
		self.slope = cvg_slope
		self.patience = patience
		self.reset()

	def reset(self):
		'''
			Resets all internal states. Equivalent of creating a new object.
		'''
		self.__epoch = 0
		self._losses = []
		self._slope_iter = 0

	def reset_slope_iter(self):
		self._slope_iter = 0

	def append(self, loss: float):
		'''
			Sums the given loss to a accumulator. Used in the minibatch loop.
		'''
		self.__epoch += loss

	def step_single(self, loss: float):
		self.append(loss)
		self.step_epoch(1)

	def step_epoch(self, dataset_size: int) -> float:
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
	def losses(self) -> List[float]:
		'''
			Recorded losses list.
		'''
		return self._losses

	def is_bound(self) -> bool:
		'''
			Returns True if lower bound is set and the last recorded loss was below or equal to it.
		'''
		return self.bound is not None and self._losses[-1] <= self.bound

	def is_converged(self) -> bool:
		'''
			Returns True if the last recorded relative slope was under the bound for beyond patience number of times.
		'''
		return self._slope_iter > self.patience
