Usage
=====

Here you can find the `source code <_modules/pythtb.html>`_ of the main PythTB module.

The main PythTB module consists of these three parts:

* :class:`pythtb.TBModel` main tight-binding model class.
* :class:`pythtb.WFArray` class for computing Berry phase (and related) properties.
* :class:`pythtb.W90` class for interface with `Wannier90 <http://www.wannier.org>`_ code that allows construction of tight-binding models based on first-principles density functional theory calculations.
* :class:`pythtb.Wannier` class for constructing Wannier functions from Bloch wavefunctions defined on a full mesh.
* :class:`pythtb.Bloch` class for working with Bloch states and related quantities.

.. automodule:: pythtb
   :undoc-members:
   :show-inheritance:
   :noindex:
   :members:

.. currentmodule:: pythtb

.. autosummary::
   :toctree: generated/
   :caption: PythTB API
   :recursive:

   TBModel
   WFArray
   W90
   Wannier
   Bloch
