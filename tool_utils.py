import time
import os
	

class Timer:
	def __init__(self):
		self.reset()

	def reset(self):
		'''
			Resets starting and lap point to current time.
		'''
		self.start_t = time.time()
		self.lap_t = self.start_t

	def total(self):
		'''
			Returns time from the recorded starting point.
		'''
		return time.time() - self.start_t

	def lap(self):
		'''
			Returns time from last recorded lap point. Modifies lap point to current point.
		'''
		t = time.time()
		d = t - self.lap_t
		self.lap_t = t
		return d

	def __format(self, dt):
		h = int(dt / 3600)
		m = int(dt / 60 % 60)
		s = dt % 60 % 60
		return '{:0>3d}:{:0>2d}:{:0>6.3f}'.format(h,m,s)

	def str_total(self):
		'''
			Returns formatted string of time from start.
		'''
		return self.__format(self.total())

	def str_lap(self):
		'''
			Returns formatted string of time from last lap and sets lap point to current.
		'''
		return self.__format(self.lap())

	def __str__(self):
		'''
			Returns formatted string of time from start.
		'''
		return self.str_total()
