from time import perf_counter


def format_time(ms:float) -> str:
	"""
		Formats the time in miliseconds into a formatted string: HHH:MM:SS.mmm
	"""
	h = int(ms / 3600)
	m = int(ms / 60 % 60)
	s = ms % 60 % 60
	return '{:0>3d}:{:0>2d}:{:0>6.3f}'.format(h,m,s)


class Timer:
	def __init__(self):
		self.reset()

	def reset(self) -> None:
		"""
			Resets starting and lap point to current time.
		"""
		self.start_t:float = perf_counter()
		self.lap_t:float = self.start_t

	@property
	def total(self) -> float:
		"""
			Returns time from the recorded starting point.
		"""
		return perf_counter() - self.start_t

	@property
	def lap(self) -> float:
		"""
			Returns time from last recorded lap point. Modifies lap point to current point.
		"""
		t = perf_counter()
		d = t - self.lap_t
		self.lap_t = t
		return d

	@property
	def lap_quiet(self) -> float:
		"""
			Returns time from last recorded lap. Doesn't modify the last lap point.
		"""
		return perf_counter() - self.lap_t

	@property
	def str_total(self) -> str:
		"""
			Returns formatted string of time from start.
		"""
		return format_time(self.total)

	@property
	def str_lap(self) -> str:
		"""
			Returns formatted string of time from last lap and sets lap point to current.
		"""
		return format_time(self.lap)

	@property
	def str_lap_quiet(self) -> str:
		"""
			Return formatted string of time from last lap, without modifying the last lap point.
		"""
		return format_time(self.lap_quiet)

	def __str__(self) -> str:
		"""
			Returns formatted string of time from start.
		"""
		return self.str_total