#!/usr/bin/env python

# demonstration of change_nonperiodic_vector function

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from __future__ import print_function
from pythtb import * # import TB model class
import numpy as np
import matplotlib.pyplot as plt

# define lattice vectors
lat=[[2.3,-0.2],[1.9,2.4]]
# define coordinates of orbitals
orb=[[0.15,0.34],[0.29,0.65]]
# make two-dimensional model
bulk_model=tb_model(2,2,lat,orb,per=[0,1])

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


# sampling of Brillouin zone
numk=21 # should be an odd number
# how many copies along direction 1
num_wire=3

# compute berry phases for the bottom band along both directions
bulk_array=wf_array(bulk_model,[numk,100]) 
bulk_array.solve_on_grid([0.,0.])
# (skip last kpoints to avoid double counting)
bulk_phase_0=np.mean(bulk_array.berry_phase([0],dir=0,contin=True)[:-1]) 
bulk_phase_1=np.mean(bulk_array.berry_phase([0],dir=1,contin=True)[:-1])
# charge center 
bulk_location=(bulk_phase_0/(2.0*np.pi))*bulk_model._lat[0]\
             +(bulk_phase_1/(2.0*np.pi))*bulk_model._lat[1]
# periodicity vectors for the wannier center
bulk_location_periodicity=[bulk_model._lat[0],bulk_model._lat[1]]

# make a better choice for the location of the charge center
# this thing is hard-coded
bulk_location=bulk_location+\
              0*bulk_location_periodicity[0]+\
              1*bulk_location_periodicity[1]

# compute what would be charge center if stack three unit cells
# on top of each other along direction 1
bulk_location_three=(bulk_location+0*bulk_location_periodicity[1])+\
                    (bulk_location+1*bulk_location_periodicity[1])+\
                    (bulk_location+2*bulk_location_periodicity[1])
# take the average of all three locations
bulk_location_three=bulk_location_three/float(num_wire)

print("Bulk model, periodic in directions 0 and 1")
print("     ---> center of three charges (with a hard-coded choice of lattice vector shift) ", bulk_location_three)
print("")

# now enlarge model along direction 1
sc_model=bulk_model.make_supercell([[1,0],[0,num_wire]],to_home=False)
sc_array=wf_array(sc_model,[numk,100]) 
sc_array.solve_on_grid([0.,0.])
# (skip last kpoints to avoid double counting)
sc_phase_0=np.mean(sc_array.berry_phase(range(num_wire),dir=0,contin=True)[:-1])
sc_phase_1=np.mean(sc_array.berry_phase(range(num_wire),dir=1,contin=True)[:-1])
# charge center
sc_location=(sc_phase_0/(2.0*np.pi))*sc_model._lat[0]\
           +(sc_phase_1/(2.0*np.pi))*sc_model._lat[1]
# periodicity vectors for the wannier center
sc_location_periodicity=[sc_model._lat[0],sc_model._lat[1]]
# make a better choice for the location of the charge center
# this thing is hard-coded
sc_location=sc_location+\
            1*sc_location_periodicity[0]+\
            2*sc_location_periodicity[1]
# divide with number of wires
sc_location=sc_location/float(num_wire)

print("Supercell of bulk model, periodic in directions 0 and 1")
print("     ---> center of total charge (with a hard-coded choice of lattice vector shift) ", sc_location)
print("")

# center of charge for system that is periodic along
# direction 0 and finite along direction 1
def get_centers_01(mod,num_bands):
    # get wavefunctions on a grid
    wfa=wf_array(mod,[numk])
    wfa.solve_on_grid([0.])
    
    # compute center of charge along the periodic direction
    ph0=wfa.berry_phase(range(num_bands),dir=0,contin=True)

    # for finite direction we simply compute average position
    pos1=[]
    for i in range(numk-1):
        # sum over three bands
        pos1.append(np.sum(wfa.position_expectation([i],range(num_bands),dir=1)))
    # average over kpoints
    pos1=np.mean(pos1)

    # construct center of charge
    ret_center=(ph0/(2.0*np.pi))*mod._lat[0]+pos1*mod._lat[1]
    # periodicity vectors for the center of charge
    ret_periodicity=[mod._lat[0]]

    return ret_center,ret_periodicity

# now cut a finite piece, so that there
# there are effectivelly only three one-dimensional "wires"
finite_model=bulk_model.cut_piece(num=num_wire, fin_dir=1, glue_edgs=False)
# get center of charge for this model
finite_location,finite_location_periodicity=get_centers_01(finite_model,num_wire)
# make a better choice for the location of the charge center
# this thing is hard-coded
finite_location=finite_location+\
                1*finite_location_periodicity[0]
#divide with number of wires
finite_location=finite_location/float(num_wire)

print("Finite model, periodic in directions 0, finite in direction 1")
print("     ---> center of total charge (with a hard-coded choice of lattice vector shift) ", finite_location)
print("")


# now create a new finite model with a different non-periodic vector
# code chooses automatically a non-periodic vector that is perpendicular to the periodic vector(s)
finite_model_orthogonalized=finite_model.change_nonperiodic_vector(np_dir=1, new_latt_vec=None)
# get center of charge for model with these periodicity vectors
finite_location_orthogonalized,finite_location_orthogonalized_periodicity=get_centers_01(finite_model_orthogonalized,num_wire)
# make a better choice for the location of the charge center
# this thing is hard-coded
finite_location_orthogonalized=finite_location_orthogonalized+\
                               5*finite_location_orthogonalized_periodicity[0]
#divide with number of wires
finite_location_orthogonalized=finite_location_orthogonalized/float(num_wire)

print("Finite model, periodic in directions 0, finite in direction 1, orthogonalized")
print("     ---> center of total charge (with a hard-coded choice of lattice vector shift) ", finite_location_orthogonalized)
print("")


# redo everything as above but with an arbitrary choice of a non-periodic vector
finite_model_arb=finite_model.change_nonperiodic_vector(np_dir=1, new_latt_vec=[-1.3,4.8])
finite_location_arb,finite_location_arb_periodicity=get_centers_01(finite_model_arb,num_wire)
# make a better choice for the location of the charge center
# this thing is hard-coded
finite_location_arb=finite_location_arb+\
                    6*finite_location_arb_periodicity[0]
#divide with number of wires
finite_location_arb=finite_location_arb/float(num_wire)

print("Finite model, periodic in directions 0, finite in direction 1, arbitrary choice of nonperiodic vector")
print("     ---> center of total charge (with a hard-coded choice of lattice vector shift) ", finite_location_arb)
print("")



# just plotting from here on

# plot all three cases
fig,axs=plt.subplots(5,1,figsize=(3.4,8.0))
for s in range(5):
    if s==0:
        case="Bulk model periodic in 0 and 1"
        loc=bulk_location_three
        st="mx"
        latt=bulk_model._lat
        peri=bulk_model._per
        orbi=bulk_model._orb
    elif s==1:
        case="Supercell of bulk model periodic in 0 and 1"
        loc=sc_location
        st="rx"
        latt=sc_model._lat
        peri=sc_model._per
        orbi=sc_model._orb
    elif s==2:
        case="Model periodic in 0, finite in 1"
        loc=finite_location
        st="gx"
        latt=finite_model._lat
        peri=finite_model._per
        orbi=finite_model._orb
    elif s==3:
        case="Model periodic in 0, finite in 1, non-periodic orthogonalized"
        loc=finite_location_orthogonalized
        st="bx"
        latt=finite_model_orthogonalized._lat
        peri=finite_model_orthogonalized._per
        orbi=finite_model_orthogonalized._orb        
    elif s==4:
        case="Model periodic in 0, finite in 1, non-periodic arbitrary"
        loc=finite_location_arb
        st="yx"
        latt=finite_model_arb._lat
        peri=finite_model_arb._per
        orbi=finite_model_arb._orb
        
    # plot lattice and orbitals
    for i in range(-8,9):
        if 1 in peri:
            use_range=range(-8,9)
        else:
            use_range=range(1)
        for j in use_range:
            path=[]
            path.append(latt[0]*(i+0)+latt[1]*(j+0))
            path.append(latt[0]*(i+1)+latt[1]*(j+0))
            path.append(latt[0]*(i+1)+latt[1]*(j+1))
            path.append(latt[0]*(i+0)+latt[1]*(j+1))
            path.append(latt[0]*(i+0)+latt[1]*(j+0))
            path=np.array(path)
            axs[s].plot(path[:,0],path[:,1],"k-",lw=0.5)
            #
            for k in range(orbi.shape[0]):
                pos=latt[0]*(orbi[k,0]+i)+latt[1]*(orbi[k,1]+j)
                axs[s].plot(pos[0],pos[1],"ko",ms=0.5)
    # plot center of charge (choice of lattice shift was hard-coded earlier)
    axs[s].plot(loc[0],loc[1],st)
    #
    axs[s].set_xlim(-1.0,18.0)
    axs[s].set_ylim(-2.0,8.0)
    axs[s].set_aspect("equal")
    axs[s].set_title(case)
    fig.tight_layout()
    fig.savefig("nonper.pdf")

