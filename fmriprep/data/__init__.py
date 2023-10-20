"""fMRIPrep data files

.. autofunction:: load

.. automethod:: load.readable

.. automethod:: load.as_path

.. automethod:: load.cached

.. autoclass:: Loader
"""
from __future__ import annotations

import atexit
import os
from contextlib import AbstractContextManager, ExitStack
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Union

try:
    from functools import cache
except ImportError:  # PY38
    from functools import lru_cache as cache

try:  # Prefer backport to leave consistency to dependency spec
    from importlib_resources import as_file, files
except ImportError:
    from importlib.resources import as_file, files  # type: ignore

try:  # Prefer stdlib so Sphinx can link to authoritative documentation
    from importlib.resources.abc import Traversable
except ImportError:
    from importlib_resources.abc import Traversable

__all__ = ["load"]


class Loader:
    """A loader for package files relative to a module

    This class wraps :mod:`importlib.resources` to provide a getter
    function with an interpreter-lifetime scope. For typical packages
    it simply passes through filesystem paths as :class:`~pathlib.Path`
    objects. For zipped distributions, it will unpack the files into
    a temporary directory that is cleaned up on interpreter exit.

    This loader accepts a fully-qualified module name or a module
    object.

    Expected usage::

        '''Data package

        .. autofunction:: load_data

        .. automethod:: load_data.readable

        .. automethod:: load_data.as_path

        .. automethod:: load_data.cached
        '''

        from fmriprep.data import Loader

        load_data = Loader(__package__)

    :class:`~Loader` objects implement the :func:`callable` interface
    and generate a docstring, and are intended to be treated and documented
    as functions.

    For greater flexibility and improved readability over the ``importlib.resources``
    interface, explicit methods are provided to access resources.

    +---------------+----------------+------------------+
    | On-filesystem | Lifetime       | Method           |
    +---------------+----------------+------------------+
    | `True`        | Interpreter    | :meth:`cached`   |
    +---------------+----------------+------------------+
    | `True`        | `with` context | :meth:`as_path`  |
    +---------------+----------------+------------------+
    | `False`       | n/a            | :meth:`readable` |
    +---------------+----------------+------------------+

    It is also possible to use ``Loader`` directly::

        from fmriprep.data import Loader

        Loader(other_package).readable('data/resource.ext').read_text()

        with Loader(other_package).as_path('data') as pkgdata:
            # Call function that requires full Path implementation
            func(pkgdata)

        # contrast to

        from importlib_resources import files, as_file

        files(other_package).joinpath('data/resource.ext').read_text()

        with as_file(files(other_package) / 'data') as pkgdata:
            func(pkgdata)

    .. automethod:: readable

    .. automethod:: as_path

    .. automethod:: cached
    """

    def __init__(self, anchor: Union[str, ModuleType]):
        self._anchor = anchor
        self.files = files(anchor)
        self.exit_stack = ExitStack()
        atexit.register(self.exit_stack.close)
        # Allow class to have a different docstring from instances
        self.__doc__ = self._doc

    @cached_property
    def _doc(self):
        """Construct docstring for instances

        Lists the public top-level paths inside the location, where
        non-public means has a `.` or `_` prefix or is a 'tests'
        directory.
        """
        top_level = sorted(
            os.path.relpath(p, self.files) + "/"[: p.is_dir()]
            for p in self.files.iterdir()
            if p.name[0] not in (".", "_") and p.name != "tests"
        )
        doclines = [
            f"Load package files relative to ``{self._anchor}``.",
            "",
            "This package contains the following (top-level) files/directories:",
            "",
            *(f"* ``{path}``" for path in top_level),
        ]

        return "\n".join(doclines)

    def readable(self, *segments) -> Traversable:
        """Provide read access to a resource through a Path-like interface.

        This file may or may not exist on the filesystem, and may be
        efficiently used for read operations, including directory traversal.

        This result is not cached or copied to the filesystem in cases where
        that would be necessary.
        """
        return self.files.joinpath(*segments)

    def as_path(self, *segments) -> AbstractContextManager[Path]:
        """Ensure data is available as a :class:`~pathlib.Path`.

        This method generates a context manager that yields a Path when
        entered.

        This result is not cached, and any temporary files that are created
        are deleted when the context is exited.
        """
        return as_file(self.files.joinpath(*segments))

    @cache
    def cached(self, *segments) -> Path:
        """Ensure data is available as a :class:`~pathlib.Path`.

        Any temporary files that are created remain available throughout
        the duration of the program, and are deleted when Python exits.

        Results are cached so that multiple calls do not unpack the same
        data multiple times, but the cache is sensitive to the specific
        argument(s) passed.
        """
        return self.exit_stack.enter_context(as_file(self.files.joinpath(*segments)))

    __call__ = cached


load = Loader(__package__)
