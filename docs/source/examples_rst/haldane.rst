Haldane Model Examples
======================
These examples illustrate the use of the Haldane model in tight-binding calculations.


.. _haldane-example: 

Haldane model
-------------

`Haldane model <http://link.aps.org/doi/10.1103/PhysRevLett.61.2015>`_
is a graphene model with complex second neighbour hoppings.

Here we have calculated the density of states as well.

.. plot :: examples/haldane/haldane.py
   :include-source:

.. _haldane_fin-example:   

Finite Haldane model
--------------------

Now let us calculate the density of states for a finite piece of the Haldane
model with and without periodic boundary conditions (PBC).

.. plot :: examples/haldane/haldane_fin.py
   :include-source:

.. _edge-example:

Edge states
-----------

Plots the edge state-eigenfunction for a finite Haldane model that
is periodic either in both directions or in only one direction.

.. plot :: examples/haldane/edge.py
   :include-source:

.. _haldane_bp-example:

Berry phases in Haldane model
-----------------------------

Calculate Berry phases along :math:`k_x` (which are proportional
to the 1D Wannier center positions along :math:`x`) as a function
of :math:`k_y` for the Haldane model.  This is done first for each
band separately, then for both together.  Two different approaches,
one less and one more automated, are illustrated.  The results
indicate that the two bands have equal and opposite Chern
numbers. Finally, the Berry curvature is calculated and printed.

.. plot :: examples/haldane/haldane_bp.py
   :include-source:

.. _haldane_hwf-example:

Hybrid Wannier centers in Haldane model
---------------------------------------

Calculates Berry phases for the Haldane model and compares it to the
hybrid Wannier centers for a ribbon of the Haldane model.

.. plot :: examples/haldane/haldane_hwf.py
   :include-source:
