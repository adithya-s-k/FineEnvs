"""`design_canvas`: the package's earlier name, kept as an alias of `designgym`.

`import design_canvas.server.app` (and any other submodule) gives the very same module object as
`designgym.server.app`, so clients, scripts and Space configs written for the old name keep working.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

import designgym as _designgym

_OLD, _NEW = "design_canvas", "designgym"


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(_OLD + "."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(_NEW + spec.name[len(_OLD):])

    def exec_module(self, module):
        pass


if not any(isinstance(f, _Alias) for f in sys.meta_path):
    sys.meta_path.insert(0, _Alias())

__path__ = []  # a package, so submodule imports reach the finder
__version__ = _designgym.__version__


def __getattr__(name):
    return getattr(_designgym, name)
