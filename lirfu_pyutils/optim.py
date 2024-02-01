import os
from typing import List, Tuple

import numpy as np
import torch


class LossTracker:
	def __init__(self, lower_bound:float=None, cvg_slope:float=1e-3, patience:int=1):
		"""
			Parameters
			----------
			lower_bound : float, optional
				Lower loss bound, used for detection of target loss. Not used if `None`. Default: `None`
			cvg_slope : float, optional
				Relative loss slope (absolute change divided by newer value), used for detection of convergence. Not used if `None`. Default: 1e-3
			patience : int, optional
				Number of iterations that forgive detections, used for ignoring noise in the loss curve. Not used if `None`. Default: 1
		"""
		self.bound = lower_bound
		self.slope = cvg_slope
		self.patience = patience
		self.reset()

	def reset(self) -> None:
		"""
			Resets all internal states. Equivalent of creating a new object.
		"""
		self.__epoch = 0
		self._losses = []
		self.__min_loss = float('inf')
		self._slope_iter = 0
		self._current_converging = False

	def reset_slope_iter(self) -> None:
		self._slope_iter = 0

	def append(self, loss:float) -> None:
		"""
			Sums the given loss to a accumulator. Used in the minibatch loop.
		"""
		self.__epoch += loss

	def step_single(self, loss:float) -> float:
		"""
			Utility method that calls `step_epoch` with dataset_size=1 to avoid averaging, e.g. if loss was averaged manually outside the tracker.
		"""
		self.append(loss)
		return self.step_epoch(1)

	def step_epoch(self, dataset_size:int) -> float:
		"""
			Averages the accumulated loss with given dataset size and evaluates the relative slope. Used in the epoch loop, after iterating minibatches.
		"""
		l = self.__epoch / dataset_size
		self.__epoch = 0
		self._losses.append(l)
		self._current_converging = (l-self.__min_loss)/(l+1e-12) > -self.slope  # Relative difference (slope).
		if self._current_converging:
			self._slope_iter += 1
		else:
			self._slope_iter = 0
		self.__min_loss = min(self.__min_loss, l)
		return l

	@property
	def losses(self) -> List[float]:
		"""
			Recorded losses list.
		"""
		return self._losses

	@property
	def is_bound(self) -> bool:
		"""
			Returns True if lower bound is set and the last recorded loss was below or equal to it.
		"""
		return self.bound is not None and self._losses[-1] <= self.bound

	@property
	def is_converged(self) -> bool:
		"""
			Returns True if the last recorded relative slope was under the bound for beyond patience number of times.
		"""
		return self._slope_iter > self.patience
	
	@property
	def is_current_converging(self) -> bool:
		"""
			Returns True if current step has triggered the convergence condition.
		"""
		return self._current_converging


class CheckpointSaver:
	"""
		Utility class for saving N best model weights and loading best.
		The intermediate weights files are deleted when obsolete.
	"""
	def __init__(self, dir:str, N:int=1):
		"""
			Parameters
			----------
			dir : str
				Directory to hold generated checkpoints.
			N : int
				Number of instances to end up with, during checkpointing overwrites the previously worst instance. If `0`, ignores checkpointing. Default: 1

		"""
		self.__dir = dir
		self.__N = N
		self.losses = np.full(N, float('inf'))
		self.checkpoints = [None,]*N

	@property
	def N(self) -> int:
		"""
			Number of models to save.
		"""
		return self.__N

	def __call__(self, dictionary:dict, loss:float, name:str) -> None:
		"""
			Stores the given dictionary if there is empty space or it is better than any previously stored by comparing the loss value.
			Name defines the filename of the checkpoint and gets appended to checkpoints directory.
		"""
		if self.__N == 0:  # Early skip.
			return
		if (self.losses >= loss).any():  # Overwrite the worst parameters.
			# Find worst.
			i = self.losses.argmax()
			self.losses[i] = loss
			# Remove worst if exists.
			if self.checkpoints[i] is not None:
				os.remove(self.checkpoints[i])
			# Save the new one.
			fp = os.path.join(self.__dir, name)
			torch.save(dictionary, fp)
			self.checkpoints[i] = fp

	def load_best(self) -> Tuple[dict,str]:
		"""
			Returns the best stored checkpoint dictionary and filepath. If number of checkpoints is `0` or no checkpoint was stored, returns (`None`,`None`)
		"""
		if self.__N == 0:  # Early skip.
			return None, None
		# Find best.
		fp = self.checkpoints[self.losses.argmin()]
		if fp is None:  # Exception, when no models were stored.
			return None, None
		return torch.load(fp), fp

	def clear(self) -> None:
		"""
			Removes all saved model instance files and directory that held them.
		"""
		for fp in self.checkpoints:
			if fp is not None:
				os.remove(fp)
		os.rmdir(self.__dir)
		self.checkpoints = None
		self.losses = None
