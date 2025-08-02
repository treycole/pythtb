Usage
=====

PythTB module consists of these primary classes:

* :class:`pythtb.TBModel` 
   Class for constructing tight-binding models and their Hamiltonians.
* :class:`pythtb.WFArray` 
   Class for storing wavefunctions on a parameter mesh, and computing Berry phases, Berry curvatures,
   Chern numbers, and other related quantities.
* :class:`pythtb.W90` 
   Class for interfacing `PythTB` with `Wannier90 <http://www.wannier.org>`_ allowing for the construction
   of tight-binding models based on first-principles density functional theory calculations.
* :class:`pythtb.Wannier` 
   Class for constructing Wannier functions from Bloch wavefunctions defined on a full mesh. These Bloch
   wavefunctions can be obtained from either tight-binding models or from first-principles calculations using the
   :class:`pythtb.W90` class. The Wannier functions' spread can then be minimized using the disentanglement and
   maximal localization algorithms implemented in the :class:`pythtb.Wannier` class.
* :class:`pythtb.Bloch` 
   Class for working with Bloch states specifically. Mimics the functionality of the
   :class:`pythtb.WFArray` class, but is focused on Bloch states and their properties.

.. currentmodule:: pythtb

.. autosummary::
   :toctree: generated/
   :caption: PythTB Classes
   :recursive:

   TBModel
   WFArray
   W90
   Wannier
   Bloch

In addition, PythTB provides a visualization module for plotting and analyzing the results obtained 
from the tight-binding models. This module includes functions for visualizing band structures, 
density of states, and the geometry of the tight-binding model. 

* :module:`pythtb.plotting` - Visualization functions for plotting band structures, density of states, and other properties.
* :module:`pythtb.models` - Predefined tight-binding models for common materials and structures.

.. autosummary::
   :toctree: generated/
   :caption: PythTB Modules
   :recursive:

   plotting
   models

.. .. automodule:: pythtb
..    :undoc-members:
..    :show-inheritance:
..    :noindex:
..    :members:



