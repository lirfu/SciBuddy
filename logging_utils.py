import os

from .tool_utils import Timer

class Logger:
	def __init__(self):
		self.debug = True

	def set_debug(self, b):
		self.debug = bool(b)

	def d(self, *msgs):
		raise RuntimeError('Not implemented')

	def i(self, *msgs):
		raise RuntimeError('Not implemented')

	def e(self, *msgs):
		raise RuntimeError('Not implemented')

	def __repr__(self):
		raise RuntimeError('Not implemented')

class StdoutLogger(Logger):
	def __init__(self, debug=True):
		self.debug = debug

	def d(self, *msgs):
		super(StdoutLogger, self).__init__()
		if self.debug:
			print(*msgs)

	def i(self, *msgs):
		print(*msgs)

	def e(self, *msgs):
		print('Err -', *msgs)

	def __repr__(self):
		return 'StdoutLogger'

class LOG:
	__instance = None
	logger = None

	def __init__(self, logger):
		LOG.logger = logger

	@staticmethod
	def init(logger=StdoutLogger()):
		if LOG.__instance is None:
			LOG.__instance = LOG(logger)
		return LOG.__instance

	@staticmethod
	def d(*msgs):
		LOG.__instance.logger.d(*msgs)

	@staticmethod
	def i(*msgs):
		LOG.__instance.logger.i(*msgs)

	@staticmethod
	def e(*msgs):
		LOG.__instance.logger.e(*msgs)

class FileLogger(Logger):
	def __init__(self, filepath, debug=True):
		self.debug = debug
		directory = os.path.dirname(filepath)
		if not directory == '':
			os.makedirs(directory, exist_ok=True)
		self.file = open(filepath, 'w')

	def __del__(self):
		self.file.close()

	def d(self, *msgs):
		if self.debug:
			self.file.write(' '.join([str(m) for m in msgs])+'\n')
			self.file.flush()

	def i(self, *msgs):
		self.file.write(' '.join([str(m) for m in msgs])+'\n')
		self.file.flush()

	def e(self, *msgs):
		self.file.write('Err - ')
		self.file.write(' '.join([str(m) for m in msgs])+'\n')
		self.file.flush()

	def __repr__(self):
		return 'FileLogger'

class MultiLogger(Logger):
	def __init__(self, *loggers):
		self.logs = list(loggers)

	def d(self, *msgs):
		for l in self.logs:
			l.d(*msgs)

	def i(self, *msgs):
		for l in self.logs:
			l.i(*msgs)

	def e(self, *msgs):
		for l in self.logs:
			l.e(*msgs)

	def __repr__(self):
		return 'MultiLogger[' + ','.join([repr(l) for l in self.logs]) + ']'

class TimedLogger(Logger):
	def __init__(self, logger):
		super(TimedLogger, self).__init__()
		self.timer = Timer()
		self.log = logger

	def d(self, *msgs):
		self.log.d(f'({self.timer.str_lap()})', *msgs)

	def i(self, *msgs):
		self.log.i(f'({self.timer.str_lap()})', *msgs)

	def e(self, *msgs):
		self.log.e(f'({self.timer.str_lap()})', *msgs)

	def __repr__(self):
		return f'TimedLogger[{repr(self.log)}]'