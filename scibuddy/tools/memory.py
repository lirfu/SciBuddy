import gc
from os import getpid
from psutil import Process


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

class MemoryMeasure:
	def __init__(self):
		self.reset()

	def reset(self):
		self.__lap = self.__start = Process(getpid()).memory_info().rss

	def total(self):
		return Process(getpid()).memory_info().rss - self.__start

	def lap(self):
		t = Process(getpid()).memory_info().rss
		d = t - self.__lap
		self.__lap = t
		return d

	def lap_quiet(self):
		return Process(getpid()).memory_info().rss - self.__lap