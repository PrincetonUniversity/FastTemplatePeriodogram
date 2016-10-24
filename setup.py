#!/usr/bin/env python

from distutils.core import setup
setup(name='pyftp',
      version=open('VERSION.txt', 'r').read().replace('\n', ''),
      description="Fast template periodogram",
      author='John Hoffman',
      author_email='jah5@princeton.edu',
      url='https://github.com/PrincetonUniversity/FastTemplatePeriodogram',
      package_dir={'pyftp' : './pyftp' },
      packages='pyftp',
      requires=[ 'numpy', 'scipy', 'pyNFFT' ],
)
