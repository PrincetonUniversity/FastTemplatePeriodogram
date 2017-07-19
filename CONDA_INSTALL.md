Installation with Conda
=======================

You can use [conda](http://conda.pydata.org) to install all the required dependencies:

The python version should be either 2.7, 3.5, or 3.6. Other versions may work,
however they haven't been tested by us.

```
$ conda create -n tempfit python=3.5
$ source activate tempfit
$ conda install numpy scipy jupyter notebook pytest pip matplotlib
$ pip install nfft
```

Next, make sure that things are working by running the unit tests:

```
$ make test
```

and finally, from the `FastTemplatePeriodogram` directory, run

```
$ python setup.py install
```
