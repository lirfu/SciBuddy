from typing import List, Dict, Callable

import signal


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
			print('[SignalCatches] Caught signal ' + signal.strsignal(signum))
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