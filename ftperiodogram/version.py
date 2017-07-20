# This is an edited copy of jvdp's `version.py` file from the `nfft`
# python package
# Version info: don't use any relative imports here, because setup.py
# runs this as a standalone script to extract the following information
from __future__ import absolute_import, division, print_function

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 1
_version_minor = 0
_version_micro = '0'  # use '' for first of series, number for 1 and above
_version_extra = ''
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = [
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6']

# Description should be a one-liner:
description = "ftperiodogram: NlogN periodic template fitting"
# Long description will go up on the pypi page
long_description = """
ftperiodogram package
=====================
`ftperiodogram` is a lightweight implementation of the fast template periodogram.
The fast template periodogram extends the [Lomb-Scargle]_ periodogram to arbitrary
signal shapes. It uses the nfft_ library to compute the non-equispaced fast Fourier
transform, and numpy_ and scipy_ libraries for other math-related computations.

For more information and links to usage examples, please see the
repository README_.

.. _README: https://github.com/PrincetonUniversity/FastTemplatePeriodogram/blob/master/README.md
.. _nfft: https://github.com/jakevdp/nfft
.. _numpy: https://www.numpy.org
.. _scipy: https://www.scipy.org
.. [Lomb-Scargle] http://docs.astropy.org/en/stable/stats/lombscargle.html

License
=======
`ftperiodogram` is licensed under the terms of the MIT license. See the file
"LICENSE.txt" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.
"""

NAME = "ftperiodogram"
MAINTAINER = "John Hoffman"
MAINTAINER_EMAIL = "jah5@princeton.edu"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/jakevdp/nfft/"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "John Hoffman, Jake VanderPlas"
AUTHOR_EMAIL = "jah5@princeton.edu, jakevdp@uw.edu"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {}
REQUIRES = ["nfft", "pytest", "scipy", "numpy"]