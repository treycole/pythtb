SSH Model Examples
====================
Below is an example demonstrating how to define a simple one-dimensional SSH model.
This example includes the source code and a plot of the resulting band structure.

.. _ssh-example:

SSH model
---------

This is a simple one-dimensional SSH model. The code plots the band structure as
a function of the intracell hopping parameter. The right panel shows the path of the vector
defining the Hamiltonian.

.. plot :: examples/ssh/ssh.py
   :include-source:


Finite chain
-------------

This finite chain example demonstrates how to define a one-dimensional SSH model with open boundary conditions.
The code plots the band structure as a function of the intracell hopping parameter, and the right panel shows 
the edge state at zero energy in the topological phase.

.. plot :: examples/ssh/finite_ssh.py
   :include-source:
