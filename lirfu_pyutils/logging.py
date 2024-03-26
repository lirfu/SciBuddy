import os
from typing import Tuple

from .tools import Timer


class Logger:
	"""
		Base class for loggers.
	"""
	def __init__(self):
		self.debug = True

	def set_debug(self, b):
		self.debug = bool(b)

	def d(self, *msgs, **kwargs):
		raise RuntimeError('Not implemented')

	def i(self, *msgs, **kwargs):
		raise RuntimeError('Not implemented')

	def e(self, *msgs, **kwargs):
		raise RuntimeError('Not implemented')

	def __repr__(self):
		raise RuntimeError('Not implemented')


class StdoutLogger(Logger):
	"""
		Logs to standard output.
	"""
	def __init__(self, debug:bool=True):
		self.debug = debug

	def d(self, *msgs, **kwargs):
		super(StdoutLogger, self).__init__()
		if self.debug:
			print(*msgs)

	def i(self, *msgs, **kwargs):
		print(*msgs)

	def e(self, *msgs, **kwargs):
		print('Err -', *msgs)

	def __repr__(self):
		return 'StdoutLogger'


class LOG:
	"""
		Global singleton logger to be re-used between different parts of code.
		Needs to be initialized to be used.
	"""
	__instance = None
	logger = None

	def __init__(self, logger:Logger):
		LOG.logger = logger

	@staticmethod
	def init(logger:Logger=StdoutLogger(), force=True):
		if force or LOG.__instance is None:
			LOG.__instance = LOG(logger)
		return LOG.__instance

	@staticmethod
	def d(*msgs, **kwargs):
		LOG.__instance.logger.d(*msgs, **kwargs)

	@staticmethod
	def i(*msgs, **kwargs):
		LOG.__instance.logger.i(*msgs, **kwargs)

	@staticmethod
	def e(*msgs, **kwargs):
		LOG.__instance.logger.e(*msgs, **kwargs)


class DevnullLogger(Logger):
	"""
		Dummy logger, just does nothing.
	"""
	def d(self, *msgs, **kwargs):
		pass

	def i(self, *msgs, **kwargs):
		pass

	def e(self, *msgs, **kwargs):
		pass

	def __repr__(self):
		return 'DevnullLogger'


class FileLogger(Logger):
	"""
		Logs into a file.
		Creates the file and missing directories.
	"""
	def __init__(self, filepath:str, debug:bool=True, append:bool=False):
		self.debug = debug
		directory = os.path.dirname(filepath)
		if directory != '':
			os.makedirs(directory, exist_ok=True)
		self.file = open(filepath, 'a' if append else 'w')

	def __del__(self):
		self.file.close()

	def d(self, *msgs, **kwargs):
		if self.debug:
			self.file.write(' '.join([str(m) for m in msgs])+'\n')
			self.file.flush()

	def i(self, *msgs, **kwargs):
		self.file.write(' '.join([str(m) for m in msgs])+'\n')
		self.file.flush()

	def e(self, *msgs, **kwargs):
		self.file.write('Err - ')
		self.file.write(' '.join([str(m) for m in msgs])+'\n')
		self.file.flush()

	def __repr__(self):
		return f'FileLogger[\'{self.file.name}\']'


class MultiLogger(Logger):
	"""
		Combines multiple loggers by notifying them using the same inputs.
	"""
	def __init__(self, *loggers:Tuple[Logger]):
		self.logs = list(loggers)

	def d(self, *msgs, **kwargs):
		for l in self.logs:
			l.d(*msgs, **kwargs)

	def i(self, *msgs, **kwargs):
		for l in self.logs:
			l.i(*msgs, **kwargs)

	def e(self, *msgs, **kwargs):
		for l in self.logs:
			l.e(*msgs, **kwargs)

	def __repr__(self):
		return 'MultiLogger[' + ','.join([repr(l) for l in self.logs]) + ']'


class TimedLogger(Logger):
	"""
		Prepends the time spent since last call.
		Contains a single timer for all call types.
	"""
	def __init__(self, logger:Logger):
		super(TimedLogger, self).__init__()
		self.timer = Timer()
		self.log = logger

	def __str(self, quiet, total):
		if quiet:
			return f'({self.timer.str_total if total else self.timer.str_lap_quiet})'
		else:
			return f'({self.timer.str_total if total else self.timer.str_lap})'

	def d(self, *msgs, quiet:bool=True, total:bool=False, **kwargs):
		self.log.d(self.__str(quiet, total), *msgs, **kwargs)

	def i(self, *msgs, quiet:bool=False, total:bool=False, **kwargs):
		self.log.i(self.__str(quiet, total), *msgs, **kwargs)

	def e(self, *msgs, quiet:bool=True, total:bool=False, **kwargs):
		self.log.e(self.__str(quiet, total), *msgs, **kwargs)

	def __repr__(self):
		return f'TimedLogger[{repr(self.log)}]'


class OffsetLogger(Logger):
	"""
		Prepends the given offset string to the inputs.
	"""
	def __init__(self, logger:Logger, off_string:str):
		super(OffsetLogger, self).__init__()
		self.log = logger
		self.offset = off_string

	def d(self, *msgs, **kwargs):
		self.log.d(self.offset, *msgs, **kwargs)

	def i(self, *msgs, **kwargs):
		self.log.i(self.offset, *msgs, **kwargs)

	def e(self, *msgs, **kwargs):
		self.log.e(self.offset, *msgs, **kwargs)

	def __repr__(self):
		return f'OffsetLogger[\'{self.offset}\'+{repr(self.log)}]'
