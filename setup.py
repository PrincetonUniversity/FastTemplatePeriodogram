#!/usr/bin/env python

import io
import os
import re

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def read(path, encoding='utf-8'):
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()


def version(path):
    """Obtain the packge version from a python file e.g. pkg/__init__.py

    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    version_file = read(path)
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = version('pyftp/__init__.py')

setup(name='pyftp',
      version=VERSION,
      description="Fast template periodogram",
      author='John Hoffman',
      author_email='jah5@princeton.edu',
      url='https://github.com/PrincetonUniversity/FastTemplatePeriodogram',
      package_dir={'pyftp' : './pyftp'},
      packages=['pyftp'],
      requires=['numpy', 'scipy', 'pynfft', 'gatspy', 'astroML', 'scikit-learn'],
)
