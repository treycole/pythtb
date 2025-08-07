#!/usr/bin/env python
# coding: utf-8

# (0d-nb)=
# # 0D tight-binding model of NH3 molecule

# In[3]:


from pythtb import TBModel  
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


# define lattice vectors
lat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# define coordinates of orbitals
sq32 = np.sqrt(3)/2
orb = [
    [(2.0 / 3.0) * sq32, 0.0, 0.0],
    [(-1.0 / 3.0) * sq32, 1.0 / 2.0, 0.0],
    [(-1.0 / 3.0) * sq32, -1.0 / 2.0, 0.0],
    [0.0, 0.0, 1.0],
]


# In[7]:


# make zero dimensional tight-binding model
my_model = TBModel(0, 3, lat, orb)

# set model parameters
delta = 0.5
t_first = 1.0

# change on-site energies so that N and H don't have the same energy
my_model.set_onsite([-delta, -delta, -delta, delta])
# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j)
my_model.set_hop(t_first, 0, 1)
my_model.set_hop(t_first, 0, 2)
my_model.set_hop(t_first, 0, 3)
my_model.set_hop(t_first, 1, 2)
my_model.set_hop(t_first, 1, 3)
my_model.set_hop(t_first, 2, 3)

print(my_model)


# Solve for the eigenenergies of the Hamiltonian

# In[9]:


evals = my_model.solve_ham()


# In[10]:


fig, ax = plt.subplots()
ax.plot(evals, "bo")
ax.set_xlim(-0.3, 3.3)
ax.set_ylim(evals.min() - 0.5, evals.max() + 0.5)
ax.set_title("Molecule levels")
ax.set_xlabel("Orbital")
ax.set_ylabel("Energy")

