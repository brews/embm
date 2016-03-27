embm
===========

A trivial global atmosphere energy-moisture balance model (EMBM) written in Python 3. Loosely based on [Fanning and Weaver 1996](http://dx.doi.org/10.1029/96JD01017).

# Requirements

* [Python 3](https://www.python.org/)
* [Numpy](http://www.numpy.org/) (embm was written with version 1.8.1)

# How do I run this?

For a complete example, see the [project's wiki page](https://github.com/brews/embm/wiki). A very simple example is below.

First, [download a copy](https://github.com/brews/embm/archive/master.zip) of the embm code. Unpackage the code and open a Python shell in the same directory as `embm.py`. In the Python shell, type:

```
import embm

m = embm.Model()
m.step(10000)
```
You can plot or just quickly checkout the fruits of your model's labor:
```
m.t[1]
m.q[1]
```

This sets up a model, which we assign to `m`. The model then runs through 10,000 time-steps, roughly 208 days in "model time" with the default settings. This is more than enough time for the model to spin up. This takes under a minute to run on my old laptop.

You can change settings and parameters or analyze variables within the model by interacting with the model instance, in this case, `m`.

For more information see the documentation within the code or [the project's humble wiki](https://github.com/brews/embm/wiki).

# What's the purpose of this?

The model, as originally described by [Fanning and Weaver 1996](http://dx.doi.org/10.1029/96JD01017), was designed as a simple atmosphere component attached to a larger ocean circulation model. It now can be used to test theoretical claims or as a simple exercise.

The model treats the atmosphere as a single-layered slab gridded into 4° x 5° cells. It accounts for simple energy (e.g. from the ocean or incoming solar radiation) and moisture exchange (e.g. crude evaporation, precipitation, and humidity) across the grid. This action is parameterized as a simple [diffusive process](http://en.wikipedia.org/wiki/Diffusion).
