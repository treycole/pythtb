Three Site Examples
===============
Below are examples of a finite one-dimensional tight-binding model
with three sites, which can be used to illustrate various concepts
in tight-binding theory, such as the Wannier center and the effect of
a one-dimensional cycle on edge states.

.. _3site_cycle-example:

One-dimensional cycle of 1D tight-binding model
-----------------------------------------------

This example considers a simple three-site one-dimensional tight-binding
model parametrized by some parameter :math:`\lambda`. As :math:`\lambda`
is changed from 0 to 1, the deepest onsite term is moved from the first
to second, then to the third, and then back to the first tight-binding
orbital. Therefore, we expect that Wannier center of the lowest band will
shift by one lattice vector as :math:`\lambda` changes from 0 to 1.

Also plots the individual on-site energies, band structure, and Wannier
center of lowest band.

.. plot :: examples/three_site/3site_cycle.py
   :include-source:


.. _3site_cycle_fin-example:

One-dimensional cycle on a finite 1D chain
------------------------------------------

This example is based on the same model as the one above but it
considers the effect of the one-dimensional cycle on the edge states
of a finite chain.

.. plot :: examples/three_site/3site_cycle_fin.py
   :include-source: