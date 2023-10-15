from time import perf_counter
from os import getpid
from psutil import Process
import gc
import colorsys
import signal
from typing import List, Dict, Callable

def arr_stats(a):
	return f'Range {(a.min(), a.max())} with shape {a.shape} and type {a.dtype}'

class Timer:
	def __init__(self):
		self.reset()

	def reset(self):
		'''
			Resets starting and lap point to current time.
		'''
		self.start_t = perf_counter()
		self.lap_t = self.start_t

	@property
	def total(self):
		'''
			Returns time from the recorded starting point.
		'''
		return perf_counter() - self.start_t

	@property
	def lap(self):
		'''
			Returns time from last recorded lap point. Modifies lap point to current point.
		'''
		t = perf_counter()
		d = t - self.lap_t
		self.lap_t = t
		return d

	@property
	def lap_quiet(self):
		'''
			Returns time from last recorded lap. Doesn't modify the last lap point.
		'''
		return perf_counter() - self.lap_t

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

class SignalCatcher:
	"""
		Catches a signal and sets the internal flag for it so user can check it (usually in a loop of something).

		The signal is caught only once, unless the catcher is reset.
		This means the second signal will pass uncaught.
	"""
	_SIGNALS:Dict[signal.Signals,bool] = {}
	_ORIG_HANDLERS:Dict[signal.Signals,List[Callable]] = {}

	def __init__(self, sig:signal.Signals=signal.SIGINT):
		self._sig = sig
		self.reset()

	def __del__(self):
		SignalCatcher.__reset_handler(self._sig)

	@staticmethod
	def __reset_handler(sig):
		if len(SignalCatcher._ORIG_HANDLERS[sig]) > 0:  # To ensure multiple calls don't raise an error.
			signal.signal(sig, SignalCatcher._ORIG_HANDLERS[sig].pop())

	@staticmethod
	def __catch(signum, frame):
		"""
			Catch the signal by setting its flag and unregistering the handler.
		"""
		if signum in SignalCatcher._SIGNALS:
			# print('lirfu_pyutils - Handling signal ' + signal.strsignal(signum))
			SignalCatcher._SIGNALS[signum] = True
			SignalCatcher.__reset_handler(signum)

	def caught(self) -> bool:
		"""
			Returns `True` only if the signal was recorded after object initialization.
		"""
		return SignalCatcher._SIGNALS[self._sig]

	def reset(self) -> None:
		"""
			Resets the internal flag for the signal and re-registeres the catching method for it.
		"""
		# Push the previous handler to the handlers stack.
		if self._sig not in SignalCatcher._ORIG_HANDLERS:
			SignalCatcher._ORIG_HANDLERS[self._sig] = []
		SignalCatcher._ORIG_HANDLERS[self._sig].append(signal.getsignal(self._sig))
		# Set the new hnadler as active.
		signal.signal(self._sig, SignalCatcher.__catch)
		SignalCatcher._SIGNALS[self._sig]= False


def get_unique_color_from_hsv(i:int, N:int) -> List[int]:
	"""
		Generate a unique HSV color for index i out of N, and convert it to RGB.
	"""
	if N > 255:
		raise RuntimeError('Supporting only 255 unique colors!')
	color = colorsys.hsv_to_rgb(i/N, 1, 1)  # NOTE: Dividing by L because color for hue 0 and 1 are the same!
	return [int(color[0]*255.99), int(color[1]*255.99), int(color[2]*255.99)]

def hex_to_rgb(v:str) -> List[int]:
	"""
		Converts a hex RGB string to a list of integers.
	"""
	return [int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)]

def rgb_to_hex(color:List[int]) -> str:
	"""
		Converts a list of integers to a hex RGB string.
	"""
	return '%0.2X'%color[0] + '%0.2X'%color[1] + '%0.2X'%color[2]