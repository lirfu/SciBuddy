import time
import gc


def arr_stats(a):
	return f'Range {(a.min(), a.max())} with shape {a.shape} and type {a.dtype}'

class Timer:
	def __init__(self):
		self.reset()

	def reset(self):
		'''
			Resets starting and lap point to current time.
		'''
		self.start_t = time.time()
		self.lap_t = self.start_t

	@property
	def total(self):
		'''
			Returns time from the recorded starting point.
		'''
		return time.time() - self.start_t

	@property
	def lap(self):
		'''
			Returns time from last recorded lap point. Modifies lap point to current point.
		'''
		t = time.time()
		d = t - self.lap_t
		self.lap_t = t
		return d

	@property
	def lap_quiet(self):
		'''
			Returns time from last recorded lap. Doesn't modify the last lap point.
		'''
		return time.time() - self.lap_t

	def __format(self, dt):
		h = int(dt / 3600)
		m = int(dt / 60 % 60)
		s = dt % 60 % 60
		return '{:0>3d}:{:0>2d}:{:0>6.3f}'.format(h,m,s)

	@property
	def str_total(self):
		'''
			Returns formatted string of time from start.
		'''
		return self.__format(self.total)

	@property
	def str_lap(self):
		'''
			Returns formatted string of time from last lap and sets lap point to current.
		'''
		return self.__format(self.lap)

	@property
	def str_lap_quiet(self):
		'''
			Return formatted string of time from last lap, without modifying the last lap point.
		'''
		return self.__format(self.lap_quiet)

	def __str__(self):
		'''
			Returns formatted string of time from start.
		'''
		return self.str_total


class GarbageCollectionContext:
	"""
		Context that forces garbage collection upon exit.
	"""
	def __init__(self, freeze=False):
		self.freeze = freeze

	def __enter__(self):
		if self.freeze:  # Freeze current stack to save from GC.
			gc.freeze()
		self.was_enabled = gc.isenabled()
		gc.enable()

	def __exit__(self, type, value, trace):
		unreachable = gc.collect(2)
		if not self.was_enabled:
			gc.disable()
		if self.freeze:
			gc.unfreeze()
