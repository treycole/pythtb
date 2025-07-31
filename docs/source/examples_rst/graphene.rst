Graphene Examples
=================

Simple models based on 2D graphene and its Dirac cone.


Graphene model
--------------

This is a toy model of a two-dimensional graphene sheet.

.. plot :: examples/graphene/graphene.py
   :include-source:
  
.. _cone-example:

Berry phase around Dirac cone in graphene
-----------------------------------------

This example computes Berry phases for a circular path (in reduced
coordinates) around the Dirac point of the graphene band structure. In
order to have a well defined sign of the Berry phase, a small on-site
staggered potential is added in order to open a gap at the Dirac point.

After computing the Berry phase around the circular loop, it also computes
the integral of the Berry curvature over a small square patch in the
Brillouin zone containing the Dirac point, and plots individual phases
for each plaquette in the array.

.. plot :: examples/graphene/cone.py
   :include-source: