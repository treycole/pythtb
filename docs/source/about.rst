About
=====

The "**Pyth**\ on **T**\ ight **B**\ inding" (``PythTB``) package is
designed to facilitate the construction and manipulation of tight-binding
models for the electronic structure of materials. It includes tools for
defining lattice structures, hopping parameters, and other model
ingredients, as well as for computing electronic properties such as
band structures and quantum geometry (Berry curvature, Berry phases,
hybrid Wannier functions, etc.). Additionally, it interfaces with
Wannier90 to allow for the construction of Wannierized tight-binding models.

The primary location for this package is at
`<http://www.physics.rutgers.edu/pythtb>`_ where the most up-to-date
releases and information can be found.

Motivations and capabilities
----------------------------

The `tight binding <http://en.wikipedia.org/wiki/Tight_binding>`_
method is an approximate approach for solving for the electronic wave
functions for electrons in solids assuming a basis of localized
atomic-like orbitals. We assume here that the orbitals are
orthonormal, and focus on the “empirical tight binding” approach in
which the Hamiltonian matrix elements are simply parametrized, as
opposed to being computed ab-initio.

The ``PythTB`` package is intended to set up and solve tight-binding
models for the electronic structure of

-  0D clusters
-  1D chains and ladders
-  2D layers (square lattice, hexagonal lattice, honeycomb lattice,
   etc.)
-  3D crystals
-  clusters, ribbons, slabs, etc., cut from higher-dimensional crystals
-  etc.

As currently written, it is not intended to handle realistic chemical
interactions. So for example, the `Slater-Koster forms
<http://en.wikipedia.org/wiki/Tight_binding#Table_of_interatomic_matrix_elements>`_
for interactions between *s*, *p* and *d* orbitals are *not currently
coded*, although the addition of such features could be considered for
a future release.

The ``PythTB`` package includes capabilities for

-  computing electron eigenvalues and eigenvectors at selected k-points
   or on a mesh of k-points
-  generating band-structure plots
-  generating density-of-states plots

It can also calculate `Berry phases, connections and curvatures
<http://en.wikipedia.org/wiki/Berry_connection_and_curvature>`_ in
ways that are useful for calculations of

-  adiabatic charge transport
-  electric polarization
-  anomalous Hall conductivity
-  topological indices
-  etc.

Finally, it provides tools for setting up more complicated
tight-binding models, e.g., by “cutting” a cluster, ribbon, or slab
out of a higher-dimensional crystal, and for visualizing the
connectivity of a tight-binding model once it has been
constructed. You can get an idea of the capabilities of the package by
browsing the :doc:`PythTB examples <examples_rst/index>`.

The code is intended for pedagogical as well as research purposes. For
example, it should be suitable for use in an undergraduate- or
graduate-level solid-state physics course as a tool for illustrating the
calculation of electronic band structructures, and it is simple enough
that it can be considered for use in homework sets or special projects
in such a course.

The ``PythTB`` package was written in Python for several reasons,
including

-  The ease of learning and using Python
-  The wide availability of Python in the community
-  The flexibility with which Python can be interfaced with graphics and
   visualization modules
-  In general, the easy extensibility of Python programs


Getting started
---------------

The best way to explore the capabilities of ``PythTB`` and to get
started in using it is to read through the :doc:`installation
instructions <install>` and :doc:`PythTB examples <examples_rst/index>`.

If you are unfamiliar with Python or are not sure whether Python and the
needed Python modules are installed on your system, see our
:doc:`python introduction <resources>` and :doc:`installation
instructions <install>`.

Note that the ``PythTB`` code is freely distributed under the terms of
the :download:`GNU GPL public license <misc/LICENSE>`. You may
use it for your own research and educational purposes, or pass it on
to others for similar use. However, the code is not guaranteed to be
bug-free, and we do not promise active support for the package.

Wannier90 interface
-------------------

Starting with Version 1.7, ``PythTB`` also provides an interface
to the `Wannier90 <http://wannier.org>`_ code, which can
be used to take the output of a first-principles density-functional
calculation and construct from it a tight-binding model, in
the basis of Wannier functions, that accurately reproduces the
first-principles bandstructure.  See :doc:`usage <usage>`.

.. _history:

History
-------

This code package had its origins in a simpler package that was
developed for use in a special-topics course on “Berry Phases in Solid
State Physics” offered by D. Vanderbilt in Fall 2010 at Rutgers
University. The students were asked to use the code as provided, or to
make extensions on their own as needed, in order to compute properties
of simple systems, such as a 2D honeycomb model of graphene, in the
tight-binding (TB) approximation. Sinisa Coh, who was a PhD student
with Vanderbilt at the time, was the initial developer and primary maintainer
of the package. Since then, many others have contributed to its development,
including those listed in the :ref:`Acknowledgments <Acknowledgments>` section.

.. _Acknowledgments:

Acknowledgments
----------------
`PythTB` has benefited from the contributions of many individuals over the years. 
Below is a list of the current maintainers and contributors, along with their affiliations.
We apologize for any omissions, and welcome feedback and corrections. 

Maintainers
^^^^^^^^^^^^^^^^
- `Trey Cole <mailto: trey@treycole.me>`_ - Rutgers University
- `David Vanderbilt <mailto: dhv@physics.rutgers.edu>`_ - Rutgers University
- `Sinisa Coh <mailto: sinisacoh@gmail.com>`_ - University of California at Riverside (formerly Rutgers University)

Contributors
^^^^^^^^^^^^^^^^
We gratefully acknowledge additional contributions to PythTB from:

- Wenshuo Liu - Reddit Inc. (formerly Rutgers University)
- Victor Alexandrov - (formerly Rutgers University)
- Tahir Yusufaly - Johns Hopkins University (formerly Rutgers University)
- Maryam Taherinejad - Hilti Group (formerly Rutgers University)

Feedback
--------

Please send comments or suggestions for improvement to `these email
addresses <mailto: trey@treycole.me, dhv@physics.rutgers.edu, sinisacoh@gmail.com>`_.

Acknowledgments and Disclaimer
------------------------------

This Web page is based in part upon work supported by the US National
Science Foundation under Grants DMR-1005838, DMR-1408838, DMR-1954856,
and DMR-2421895.  Any opinions, findings, and
conclusions or recommendations expressed in this material are those of
the author and do not necessarily reflect the views of the National
Science Foundation.
