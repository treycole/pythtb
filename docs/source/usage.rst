Usage
=====

Here you can find the `source code <https://github.com/treycole/pythtb>`_ of the main PythTB module.

The main PythTB module consists of these three parts:

* :class:`pythtb.TBModel` - main tight-binding model class.
* :class:`pythtb.WFArray` - class for storing wavefunctions on a parameter mesh, and computing Berry phases, Berry curvatures,
   Chern numbers, and other related quantities.
* :class:`pythtb.W90` - class for interfacing `PythTB` with `Wannier90 <http://www.wannier.org>`_ allowing for the construction
    of tight-binding models based on first-principles density functional theory calculations.
* :class:`pythtb.Wannier` - class for constructing Wannier functions from Bloch wavefunctions defined on a full mesh. These Bloch
   wavefunctions can be obtained from either tight-binding models or from first-principles calculations using the
   :class:`pythtb.W90` class. The Wannier functions' spread can then be minimized using the disentanglement and 
   maximal localization algorithms implemented in the :class:`pythtb.Wannier` class. 
* :class:`pythtb.Bloch` - class for working with Bloch states specifically. Mimics the functionality of the
   :class:`pythtb.WFArray` class, but is focused on Bloch states and their properties.

API Summary
------------
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

.. .. automodule:: pythtb
..    :undoc-members:
..    :show-inheritance:
..    :noindex:
..    :members:



