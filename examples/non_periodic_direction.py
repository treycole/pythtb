#!/usr/bin/env python

# demonstration of change_nonperiodic_vector function

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from __future__ import print_function
from pythtb import * # import TB model class
from pythtb import _one_phase_cont
import numpy as np
import matplotlib.pyplot as plt

# define lattice vectors
lat=[[2.3,-0.5],[0.3,2.6]]
# define coordinates of orbitals
orb=[[0.15,0.41],[0.29,0.65]]
# make two-dimensional model
bulk_model=tb_model(2,2,lat,orb)

# Add hopping terms.  Note that there are no hoppings
# along the second periodic lattice vector.  Therefore
# this nominally two-dimensional material is just an
# infinite stack of one-dimensional wires.
#
t_first=0.8+0.6j
t_second=2.0
delta=-0.8
bulk_model.set_onsite([-delta,delta])
bulk_model.set_hop(t_second, 0, 0, [1,0])
bulk_model.set_hop(t_second, 1, 1, [1,0])
bulk_model.set_hop(t_first, 0, 1, [0,0])
bulk_model.set_hop(t_first, 1, 0, [1,0])

# compute berry phases for the bottom band along both directions
numk=11 # should be odd number
bulk_array=wf_array(bulk_model,[numk,2]) 
bulk_array.solve_on_grid([0.,0.])
#
bulk_phase_0=np.mean(bulk_array.berry_phase([0],dir=0,contin=True)[:-1])
bulk_phase_1=np.mean(bulk_array.berry_phase([0],dir=1,contin=True)[:-1])

# SKIP LAST ONE

print("Bulk: ",(bulk_phase_0/(2.0*np.pi))*bulk_model._lat[0]+
               (bulk_phase_1/(2.0*np.pi))*bulk_model._lat[1])


# now cut a finite piece, so that there
# there is effectivelly only one one-dimensional "wire".
finite_model=bulk_model.cut_piece(num=1, fin_dir=1, glue_edgs=False)

# now compute berry phase for finite_model
finite_array=wf_array(finite_model,[numk])
finite_array.solve_on_grid([0.])
#
finite_phase_0=finite_array.berry_phase([0],dir=0,contin=True)
# for finite direction we can simply compute
# expectation of the position operator
finite_position_1=[]
for i in range(numk-1):
    finite_position_1.append(finite_array.position_expectation([i],[0],dir=1))
finite_position_1=np.array(finite_position_1)
finite_position_1[0]=no_2pi(finite_position_1[0]*2.0*np.pi,0.0)/(2.0*np.pi)
#print(_one_phase_cont(finite_position_1*2.0*np.pi,finite_position_1[0]*2.0*np.pi)/(2.0*np.pi))
finite_position_1=np.mean(_one_phase_cont(finite_position_1*2.0*np.pi,finite_position_1[0]*2.0*np.pi)/(2.0*np.pi))

finite_location=(finite_phase_0/(2.0*np.pi))*finite_model._lat[0]+\
                  finite_position_1          *finite_model._lat[1]
print("FINITE: ",finite_location)


# now create a new finite model in which non-periodic vector is made
# perpendicular to the periodic vector.
finite_model_orthogonalized=finite_model.change_nonperiodic_vector(np_dir=1, new_latt_vec=None, to_home=False)

# Now repeat the same steps as before, but this time for a
# model with orthogonalized vectors
finite_array_orthogonalized=wf_array(finite_model_orthogonalized,[numk])
finite_array_orthogonalized.solve_on_grid([0.])
#
finite_phase_orthogonalized_0=finite_array_orthogonalized.berry_phase([0],dir=0,contin=True)
# for finite direction we can simply compute
# expectation of the position operator
finite_position_orthogonalized_1=[]
for i in range(numk-1):
    finite_position_orthogonalized_1.append(finite_array_orthogonalized.position_expectation([i],[0],dir=1))
finite_position_orthogonalized_1=np.array(finite_position_orthogonalized_1)
finite_position_orthogonalized_1[0]=no_2pi(finite_position_orthogonalized_1[0]*2.0*np.pi,0.0)/(2.0*np.pi)
#print(_one_phase_cont(finite_position_orthogonalized_1*2.0*np.pi,finite_position_orthogonalized_1[0]*2.0*np.pi)/(2.0*np.pi))
finite_position_orthogonalized_1=np.mean(_one_phase_cont(finite_position_orthogonalized_1*2.0*np.pi,finite_position_orthogonalized_1[0]*2.0*np.pi)/(2.0*np.pi))

finite_location_orthogonalized=(finite_phase_orthogonalized_0/(2.0*np.pi))*finite_model_orthogonalized._lat[0]+\
                                 finite_position_orthogonalized_1          *finite_model_orthogonalized._lat[1]
print("FINITE_orthogonalized: ",finite_location_orthogonalized)



(fig,ax)=finite_model.visualize(0,1)
for i in range(10):
    ax.plot(finite_location[0],finite_location[1],"ks")
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
fig.savefig("b.pdf")
(fig,ax)=finite_model_orthogonalized.visualize(0,1)
for i in range(10):
    ax.plot(finite_location_orthogonalized[0],finite_location_orthogonalized[1],"ks")
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
fig.savefig("c.pdf")
