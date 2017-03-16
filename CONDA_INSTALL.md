Installation with Conda
=======================

You can use [conda](http://conda.pydata.org) to install all the required dependencies:

```
$ conda create -n fast-template-periodogram python=3.5
$ source activate fast-template-periodogram
$ conda install jupyter notebook scipy astropy pytest
$ conda install pynfft -c conda-forge
$ pip install gatspy
```

Next, make sure that things are working by running the unit tests:

```
$ make test
```