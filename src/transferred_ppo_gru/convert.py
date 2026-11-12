"""Compatibility import; implementation lives in arbiter_miroshnik.research.convert."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("arbiter_miroshnik.research.convert")
