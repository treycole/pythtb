Hybrid Wannier functions Examples
================================
This section contains examples of how to use the
:class:`pythtb.WFArray` class to work with hybrid Wannier functions
and related calculations.

.. _cubic_slab_hwf:

Hybrid Wannier functions on a cubic slab
----------------------------------------

This code illustrates a calculation of the Berry phases along x
of individual z-localized hybrid Wannier bands for a slab model
(finite in z but extended in x and y), using a *wf_array* structure
to simplify the calculation.

.. plot :: examples/slab/cubic_slab_hwf.py
   :include-source: