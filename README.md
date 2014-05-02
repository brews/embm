embm
===========

A trivial global atmosphere energy-moisture balance model (EMBM) written in Python 3. Loosely based on [Fanning and Weaver 1996](http://dx.doi.org/10.1029/96JD01017).

# Requirements

* [Python version 3](https://www.python.org/)
* [Numpy](http://www.numpy.org/)
* [Matplotlib](http://matplotlib.org/) (for optional plotting)

# How do I run this?

First, [download a copy](https://github.com/brews/embm/archive/master.zip) of the embm code. Unpackage the code and open a Python shell in the same directory as `embm.py`. In the Python shell, type:

```
import embm

m = embm.Model()
m.step(10000, verbose = True)
```

This sets up a model, which we assign to `m`. The model then runs through 10,000 time-steps, roughly 208 days in "model time" with the default settings. This is more than enough time for the model to spin up.

The `verbose = True` argument tells the program to print a progress bar to the shell.

You can change settings and parameters or analyze variables within the model by interacting with the model instance, in this case, `m`.

For more information see the documentation within the code or [the project's humble wiki](https://github.com/brews/embm/wiki).

# What's the purpose of this?

The model, as originally described by [Fanning and Weaver 1996](http://dx.doi.org/10.1029/96JD01017), was designed to acts as a simple atmosphere component attached to a larger ocean circulation model. Now, it might be used to test theoretical claims or as a simple exercise.

The model treats the atmosphere as a single-layered slab gridded into 4° x 5° cells. It accounts for simple energy (e.g. from the ocean or incoming solar radiation) and moisture exchange (e.g. crude evaporation, precipitation, and humidity) across the grid. This action is parameterized as a simple [diffusive process](http://en.wikipedia.org/wiki/Diffusion).
