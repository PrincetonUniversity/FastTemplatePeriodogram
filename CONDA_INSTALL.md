Installation with Conda
=======================

You can use [conda](http://conda.pydata.org) to install all the required dependencies:

```
$ conda create -n tempfit python=3.5
$ source activate tempfit
$ conda install jupyter notebook scipy astropy pytest matplotlib
$ conda install pynfft -c conda-forge
```

Next, make sure that things are working by running the unit tests:

```
$ make test
```

and finally, from the `FastTemplatePeriodogram` directory, run

```
$ python setup.py install
```
