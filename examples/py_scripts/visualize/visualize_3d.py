from pythtb.models import fu_kane_mele

# Reference Model
t = 1  # spin-independent first-neighbor hop
soc = 1  # spin-dependent second-neighbor hop
m = 1  # magnetic field magnitude
beta = 1  # Adiabatic parameter
fkm_model = fu_kane_mele(t, soc, m, beta)

fig = fkm_model.visualize_3d(draw_hoppings=True) 
fig.show()
