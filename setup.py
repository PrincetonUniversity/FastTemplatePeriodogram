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


VERSION = version('ftperiodogram/__init__.py')

setup(name='ftperiodogram',
      version=VERSION,
      description="Tools for efficiently fitting periodic templates to irregularly-sampled timeseries data",
      author='John Hoffman, Jake Vanderplas',
      author_email='jah5@princeton.edu, jakevdp@cs.washington.edu',
      url='https://github.com/PrincetonUniversity/FastTemplatePeriodogram',
      packages=['ftperiodogram',
                'ftperiodogram.tests'],
      requires=['numpy', 'scipy', 'nfft'],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'],
     )
