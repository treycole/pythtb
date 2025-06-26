from .utils import _is_int, _offdiag_approximation_warning_and_stop
from .tb_model import TBModel
from .k_mesh import KMesh
import numpy as np
import copy  # for deepcopying
from itertools import product

__all__ = ["WFArray", "Bloch"]


def _wf_dpr(wf1, wf2):
    """calculate dot product between two wavefunctions.
    wf1 and wf2 are of the form [orbital,spin]"""
    return np.dot(wf1.flatten().conjugate(), wf2.flatten())


def _one_berry_loop(wf, berry_evals=False):
    """Do one Berry phase calculation (also returns a product of M
    matrices).  Always returns numbers between -pi and pi.  wf has
    format [kpnt,band,orbital,spin] and kpnt has to be one dimensional.
    Assumes that first and last k-point are the same. Therefore if
    there are n wavefunctions in total, will calculate phase along n-1
    links only!  If berry_evals is True then will compute phases for
    individual states, these corresponds to 1d hybrid Wannier
    function centers. Otherwise just return one number, Berry phase."""
    # number of occupied states
    nocc = wf.shape[1]
    # temporary matrices
    prd = np.identity(nocc, dtype=complex)
    ovr = np.zeros([nocc, nocc], dtype=complex)
    # go over all pairs of k-points, assuming that last point is overcounted!
    for i in range(wf.shape[0] - 1):
        # generate overlap matrix, go over all bands
        for j in range(nocc):
            for k in range(nocc):
                ovr[j, k] = _wf_dpr(wf[i, j, :], wf[i + 1, k, :])
        # only find Berry phase
        if not berry_evals:
            # multiply overlap matrices
            prd = np.dot(prd, ovr)
        # also find phases of individual eigenvalues
        else:
            # cleanup matrices with SVD then take product
            matU, sing, matV = np.linalg.svd(ovr)
            prd = np.dot(prd, np.dot(matU, matV))
    # calculate Berry phase
    if not berry_evals:
        det = np.linalg.det(prd)
        pha = (-1.0) * np.angle(det)
        return pha
    # calculate phases of all eigenvalues
    else:
        evals = np.linalg.eigvals(prd)
        eval_pha = (-1.0) * np.angle(evals)
        # sort these numbers as well
        eval_pha = np.sort(eval_pha)
        return eval_pha

def _one_flux_plane(wfs2d):
    "Compute fluxes on a two-dimensional plane of states."
    # size of the mesh
    nk0 = wfs2d.shape[0]
    nk1 = wfs2d.shape[1]
    # number of bands (will compute flux of all bands taken together)
    nbnd = wfs2d.shape[2]

    # here store flux through each plaquette of the mesh
    all_phases = np.zeros((nk0 - 1, nk1 - 1), dtype=float)

    # go over all plaquettes
    for i in range(nk0 - 1):
        for j in range(nk1 - 1):
            # generate a small loop made out of four pieces
            wf_use = []
            wf_use.append(wfs2d[i, j])
            wf_use.append(wfs2d[i + 1, j])
            wf_use.append(wfs2d[i + 1, j + 1])
            wf_use.append(wfs2d[i, j + 1])
            wf_use.append(wfs2d[i, j])
            wf_use = np.array(wf_use, dtype=complex)
            # calculate phase around one plaquette
            all_phases[i, j] = _one_berry_loop(wf_use)

    return all_phases


class WFArray:
    r"""

    This class is used to store and manipulate an array of
    wavefunctions of a tight-binding model
    :class:`pythtb.TBModel` on a regular or non-regular grid
    These are typically the Bloch energy eigenstates of the
    model, but this class can also be used to store a subset
    of Bloch bands, a set of hybrid Wannier functions for a
    ribbon or slab, or any other set of wavefunctions that
    are expressed in terms of the underlying basis orbitals.
    It provides methods that can be used to calculate Berry
    phases, Berry curvatures, 1st Chern numbers, etc.

    *Regular k-space grid*:
    If the grid is a regular k-mesh (no parametric dimensions),
    a single call to the function
    :func:`pythtb.WFArray.solve_on_grid` will both construct a
    k-mesh that uniformly covers the Brillouin zone, and populate
    it with wavefunctions (eigenvectors) computed on this grid.
    The last point in each k-dimension is set so that it represents
    the same Bloch function as the first one (this involves the
    insertion of some orbital-position-dependent phase factors).

    Example :ref:`haldane_bp-example` shows how to use WFArray on
    a regular grid of points in k-space. Examples :ref:`cone-example`
    and :ref:`3site_cycle-example` show how to use non-regular grid of
    points.

    *Parametric or irregular k-space grid grid*:
    An irregular grid of points, or a grid that includes also
    one or more parametric dimensions, can be populated manually
    with the help of the *[]* operator.  For example, to copy
    eigenvectors *evec* into coordinate (2,3) in the *WFArray*
    object *wf* one can simply do::

      wf[2,3]=evec

    The wavefunctions (here the eigenvectors) *evec* above
    are expected to be in the format *evec[state,orbital]*
    (or *evec[state,orbital,spin]* for the spinfull calculation),
    where *state* typically runs over all bands.
    This is the same format as returned by
    :func:`pythtb.TBModel.solve_one` or
    :func:`pythtb.TBModel.solve_all` (in the latter case one
    needs to restrict it to a single k-point as *evec[:,kpt,:]*
    if the model has *dim_k>=1*).

    If WFArray is used for closed paths, either in a
    reciprocal-space or parametric direction, then one needs to
    include both the starting and ending eigenfunctions even though
    they are physically equivalent.  If the array dimension in
    question is a k-vector direction and the path traverses the
    Brillouin zone in a primitive reciprocal-lattice direction,
    :func:`pythtb.WFArray.impose_pbc` can be used to associate
    the starting and ending points with each other; if it is a
    non-winding loop in k-space or a loop in parameter space,
    then :func:`pythtb.WFArray.impose_loop` can be used instead.
    (These may not be necessary if only Berry fluxes are needed.)

    Example :ref:`3site_cycle-example` shows how one
    of the directions of *WFArray* object need not be a k-vector
    direction, but can instead be a Hamiltonian parameter :math:`\lambda`
    (see also discussion after equation 4.1 in :download:`notes on
    tight-binding formalism <misc/pythtb-formalism.pdf>`).

    The wavevectors stored in *WFArray* are typically Hamiltonian
    eigenstates (e.g., Bloch functions for k-space arrays),
    with the *state* index running over all bands.  However, a
    *WFArray* object can also be used for other purposes, such
    as to store only a restricted set of Bloch states (e.g.,
    just the occupied ones); a set of modified Bloch states
    (e.g., premultiplied by a position, velocity, or Hamiltonian
    operator); or for hybrid Wannier functions (i.e., eigenstates
    of a position operator in a nonperiodic direction).  For an
    example of this kind, see :ref:`cubic_slab_hwf`.

    :param model: Object of type :class:`pythtb.TBModel` representing
      tight-binding model associated with this array of eigenvectors.

    :param mesh_size: List of dimensions of the mesh of the *WFArray*,
      in order of reciprocal-space and/or parametric directions.

    :param nstates: Optional parameter specifying the number of states
      packed into the *WFArray* at each point on the mesh.  Defaults
      to all states (i.e., norb*nspin).

    Example usage::

      # Construct WFArray capable of storing an 11x21 array of
      # wavefunctions
      wf = WFArray(tb, [11, 21])
      # populate this WFArray with regular grid of points in
      # Brillouin zone
      wf.solve_on_grid([0.0, 0.0])

      # Compute set of eigenvectors at one k-point
      (eval, evec) = tb.solve_one([kx, ky], eig_vectors = True)
      # Store it manually into a specified location in the array
      wf[3,4] = evec
      # To access the eigenvectors from the same position
      print(wf[3,4])

    """

    def __init__(self, model: TBModel, mesh_size: list|tuple, nstates=None):
        #TODO: We would like to have a KMesh object associated with the WFArray
        # this way we can store information about the k-points corresponding to each
        # point in the WFArray, and also the k-points can be used to impose PBC automatically.
        # To do this, the user needs to specify the k-points when constructing the WFArray. 
        # Some dimensions of the mesh may be adiabatic parameters, or paths in k-space. Somehow
        # this should be distinguished from the regular k-mesh. 

        # check that model is of type TBModel
        if not isinstance(model, TBModel):
            raise TypeError("model must be of type TBModel")
        # check that mesh_size is a list or tuple
        if not (isinstance(mesh_size, list) or isinstance(mesh_size, tuple)):
            raise TypeError("mesh_size must be a list or tuple")
        # check that mesh_size is not empty
        if len(mesh_size) == 0:
            raise ValueError("mesh_size must not be empty")
        # check that mesh_size contains only integers
        if not all(_is_int(x) for x in mesh_size):
            raise TypeError("mesh_size must contain only integers")


        # number of electronic states for each k-point
        if nstates is None:
            self._nstates = model.nstate  # this = norb*nspin = no. of bands
            # note: 'None' means to use the default, which is all bands!
        else:
            if not _is_int(nstates):
                raise Exception("\n\nArgument nstates not an integer")
            self._nstates = nstates  # set by optional argument
        # number of spin components
        self._nspin = model.nspin
        # number of orbitals
        self._norb = model.norb
        # store orbitals from the model
        self._orb = model.orb_vecs
        # store entire model as well
        self._model = model
        # store dimension of array of points on which to keep wavefunctions
        self._mesh_size = np.array(mesh_size)
        self._dim_mesh = len(self._mesh_size)
        # all dimensions should be 2 or larger, because pbc can be used
        if True in (self._mesh_size <= 1).tolist():
            raise Exception(
                "\n\nDimension of WFArray object in each direction must be 2 or larger."
            )
        self._pbc_axes = []  # axes along which periodic boundary conditions are imposed
        self._loop_axes = []  # axes along which loops are imposed
        # generate temporary array used later to generate object ._wfs
        wfs_dim = np.copy(self._mesh_size)
        wfs_dim = np.append(wfs_dim, self._nstates)
        wfs_dim = np.append(wfs_dim, self._norb)
        if self._nspin == 2:
            wfs_dim = np.append(wfs_dim, self._nspin)
        # store wavefunctions in the form
        #   _wfs[kx_index,ky_index, ... ,state,orb,spin]
        self._wfs = np.zeros(wfs_dim, dtype=complex)
    
    def __getitem__(self, key):
        self._check_key(key)
        return self._wfs[key]

    def __setitem__(self, key, value):
        self._check_key(key)
        self._wfs[key] = np.array(value, dtype=complex)

    def _check_key(self, key):
        # key is an index list specifying the grid point of interest
        if self._dim_mesh == 1:
            if isinstance(key, (tuple, list, np.ndarray)):
                assert len(key) == 1, "Key should be an integer or a tuple of length 1!"
                key = key[0]  # convert to integer
            elif not isinstance(key, (int, np.integer)):
                raise TypeError("Key should be an integer!")
            if key < (-1) * self._mesh_size[0] or key >= self._mesh_size[0]:
                raise IndexError("Key outside the range!")
        else:
            if len(key) != self._dim_mesh:
                raise TypeError("Wrong dimensionality of key!")
            for i, k in enumerate(key):
                if not _is_int(k):
                    raise TypeError("Key should be set of integers!")
                if k < (-1) * self._mesh_size[i] or k >= self._mesh_size[i]:
                    raise IndexError("Key outside the range!")

    @property
    def wfs(self):
        r"""Returns the wavefunctions stored in the *WFArray* object."""
        return self._wfs
    
    @property
    def shape(self):
        r"""Returns the shape of the wavefunction array."""
        return self._wfs.shape
    
    @property
    def mesh_size(self):
        r"""Returns the mesh dimensions of the *WFArray* object."""
        return self._mesh_size
    
    @property
    def dim_mesh(self):
        r"""Returns the number of dimensions of the *WFArray* object."""
        return self._dim_mesh
    
    @property
    def nstates(self):
        r"""Returns the number of states (or bands) stored in the *WFArray* object."""
        return self._nstates
    
    @property
    def nspin(self):
        r"""Returns the number of spin components stored in the *WFArray* object."""
        return self._nspin
    
    @property
    def norb(self):
        r"""Returns the number of orbitals stored in the *WFArray* object."""
        return self._norb
    
    @property
    def model(self):
        r"""Returns the underlying TBModel object associated with the *WFArray*."""
        return self._model
    
    @property
    def param_path(self):
        r"""Returns the parameter path (e.g., k-points) along which the model was solved.
        This is only set if the model was solved along a path using `solve_on_path`."""
        return getattr(self, '_param_path', None)
    

    def get_states(self, flatten_spin=False):
        """Returns dictionary containing Bloch and cell-periodic eigenstates."""
        # shape is [nk1, ..., nkd, [n_lambda,] n_state, n_orb[, n_spin]
        wfs = self.wfs

        # flatten last two axes together to condense spin and orbital indices
        if flatten_spin and self.nspin == 2:
            wfs = wfs.reshape((*wfs.shape[:-2], -1))

        return wfs


    def get_bloch_states(self, flatten_spin=False):
        """Returns Bloch states from the WFArray."""
        # shape is [nk1, ..., nkd, [n_lambda,] n_state, n_orb[, n_spin]
        u_wfs = self.wfs

        # make sure that u_wfs is consistent with being defined on a regular k-mesh
        if self.dim_mesh != self.model.dim_k:
            raise ValueError(
                f"WFArray is initialized for a mesh of dimensions {self.dim_mesh}, "
                f"but the model has dim_k = {self.model.dim_k}. Bloch states assumes the"
                f"WFArray is defined on a regular k-mesh."
            )
        
        psi_wfs = self._apply_phase(wfs=u_wfs, inverse=False)

        # flatten last two axes together to condense spin and orbital indices
        if flatten_spin:
            u_wfs = u_wfs.reshape((*u_wfs.shape[:-2], -1))
            psi_wfs = psi_wfs.reshape((*psi_wfs.shape[:-2], -1))

        return_states = {
            'u_wfs': u_wfs,
            'psi_wfs': psi_wfs,
        }
        return return_states

    def solve_on_path(self, k_arr):
        """
        Solve the model along a 1D parameter path (e.g., k-points).
        Stores eigenvectors and eigenvalues along this path.

        Parameters
        ----------
        k_arr : array-like, shape (n_points, dim_k)
            Sequence of points (e.g., k-points) at which to solve the model.
            Must match the model's dim_k and the WFArray's mesh_size[0].

        Returns
        -------
        energies : ndarray, shape (n_points, nstates)
            Eigenvalues at each point along the path.
        wfs : ndarray, shape (n_points, nstates, norb[, nspin])
            Corresponding eigenvectors stored in the WFArray.
        """
        k_arr = np.asarray(k_arr, dtype=float)
        # check dimensionality
        if k_arr.ndim != 2 or k_arr.shape[1] != self.model.dim_k:
            raise ValueError(
                f"Expected k_arr to have shape (n_points, {self.model.dim_k}), "
                f"but got shape {k_arr.shape}."
            )
        n_points = k_arr.shape[0]
        # check that the number of points matches the mesh size
        if self.dim_mesh != 1 or self.mesh_size[0] != n_points:
            raise ValueError(
                f"WFArray is initialized for a mesh size of {self.mesh_size[0]}, "
                f"but k_arr has {n_points} points."
            )
        self._param_path = k_arr.copy()

        eigvals, eigvecs = self.model.solve_ham(k_arr, return_eigvecs=True)
        for idx, pt in enumerate(k_arr):
            self.energies[idx] = eigvals[idx]
            self._wfs[(idx,)] =  eigvecs[idx]


    #TODO: Clarify the role of start_k. When would it be anything other than [0, 0] 
    # or [-0.5, -0.5]? 
    def solve_on_grid(self, start_k):
        r"""

        Solve a tight-binding model on a regular mesh of k-points covering
        the entire reciprocal-space unit cell. Both points at the opposite
        sides of reciprocal-space unit cell are included in the array.

        This function also automatically imposes periodic boundary
        conditions on the eigenfunctions. See also the discussion in
        :func:`pythtb.WFArray.impose_pbc`.

        :param start_k: Origin of a regular grid of points in the reciprocal space.

        :returns:
          * **gaps** -- returns minimal direct bandgap between n-th and n+1-th
              band on all the k-points in the mesh.  Note that in the case of band
              crossings one may have to use very dense k-meshes to resolve
              the crossing.

        Example usage::

          # Solve eigenvectors on a regular grid anchored
          # at a given point
          wf.solve_on_grid([-0.5, -0.5])

        """
        # check dimensionality
        if self.dim_mesh != self._model._dim_k:
            raise Exception(
                "\n\nIf using solve_on_grid method, dimension of WFArray must equal"
                "\ndim_k of the tight-binding model!"
            )

        # check number of states
        if self.nstates != self.model.nstate:
            raise Exception(
                "\n\nWhen initializing this object, you specified nstates to be "
                + str(self.nstates)
                + ", but"
                "\nthis does not match the total number of bands specified in the model,"
                "\nwhich was "
                + str(self.model.nstate)
                + ".  If you wish to use the solve_on_grid method, do"
                "\nnot specify the nstates parameter when initializing this object.\n\n"
            )

        # store start_k
        self._start_k = start_k

        # to return gaps at all k-points
        if self.nstates <= 1:
            all_gaps = None  # trivial case since there is only one band
        else:
            gap_dim = np.copy(self.mesh_size) - 1
            gap_dim = np.append(gap_dim, self.nstates - 1)
            all_gaps = np.zeros(gap_dim, dtype=float)

        k_pts = [
            np.linspace(start_k[idx], start_k[idx] + 1, nk-1, endpoint=False)
            for idx, nk in enumerate(self.mesh_size)
        ]
        k_pts_sq = np.stack(np.meshgrid(*k_pts, indexing="ij"), axis=-1)
        k_pts = k_pts_sq.reshape(-1, self.dim_mesh)

        evals, evecs = self._model.solve_ham(k_pts, return_eigvecs=True)

        # reshape to square mesh: (nk-1, nk-1, ..., nk-1, nstate) for evals
        evals = evals.reshape(tuple([nk-1 for nk in self.mesh_size]) + evals.shape[1:])
        # reshape to square mesh: (nk-1, nk-1, ..., nk-1, nstate, nstate) for evecs
        evecs = evecs.reshape(tuple([nk-1 for nk in self.mesh_size]) + evecs.shape[1:])

        # set gaps
        all_gaps = evals[..., 1:] - evals[..., :-1]

        # mapping from 1d index to multi-dimensional index
        axes = [np.arange(nk-1) for nk in self.mesh_size]
        idx_arr = np.array(np.meshgrid(*axes, indexing='ij'))
        idx_arr = idx_arr.reshape(idx_arr.shape[0], -1).T
        idx_arr = np.array(idx_arr, dtype=int)

        # set wavefunctions in the array
        for idx in idx_arr:
            self[*idx] = evecs[*idx]

        # impose periodic boundary conditions along all directions
        for dir in range(self.dim_mesh):
            # impose periodic boundary conditions
            self.impose_pbc(dir, self.model.per[dir])

        if all_gaps is not None:
            return all_gaps.min(axis=tuple(range(self.dim_mesh)))
        else:
            return None


    def solve_on_one_point(self, kpt, mesh_indices):
        r"""

        Solve a tight-binding model on a single k-point and store the eigenvectors
        in the *WFArray* object in the location specified by *mesh_indices*.

        :param kpt: List specifying desired k-point

        :param mesh_indices: List specifying associated set of mesh indices

        :returns:
          None

        Example usage::

          # Solve eigenvectors on a sphere of radius kappa surrounding
          # point k_0 in 3d k-space and pack into a predefined 2d WFArray
          for i in range[n+1]:
            for j in range[m+1]:
              theta=np.pi*i/n
              phi=2*np.pi*j/m
              kx=k_0[0]+kappa*np.sin(theta)*np.cos(phi)
              ky=k_0[1]+kappa*np.sin(theta)*np.sin(phi)
              kz=k_0[2]+kappa*np.cos(theta)
              wf.solve_on_one_point([kx,ky,kz],[i,j])

        """

        _, evec = self.model.solve_ham(kpt, return_eigvecs=True)
        if _is_int(mesh_indices):
            self._wfs[(mesh_indices,)] = evec
        else:
            self._wfs[tuple(mesh_indices)] = evec

    #TODO: This function should be removed or modified
    # it does not preserve the proper nstates
    def choose_states(self, subset):
        r"""

        Create a new *WFArray* object containing a subset of the
        states in the original one.

        :param subset: List of integers specifying states to keep

        :returns:
          * **wf_new** -- returns a *WFArray* that is identical in all
              respects except that a subset of states have been kept.

        Example usage::

          # Make new *WFArray* object containing only two states
          wf_new=wf.choose_states([3,5])

        """

        # make a full copy of the WFArray
        wf_new = copy.deepcopy(self)

        subset = np.array(subset, dtype=int)
        if subset.ndim != 1:
            raise Exception("\n\nParameter subset must be a one-dimensional array.")

        wf_new._nstates = subset.shape[0]
        if self._model.nspin == 2:
            wf_new._wfs = wf_new._wfs[..., subset, :, :]
        elif self._model.nspin == 1:
            wf_new._wfs = wf_new._wfs[..., subset, :]
        else:   
            raise Exception(
                "\n\nWFArray object can only handle spinless or spin-1/2 models."
            )

        return wf_new

    #TODO: Same as above, this function should be removed or modified
    # it does not preserve the proper nstates
    def empty_like(self, nstates=None):
        r"""

        Create a new empty *WFArray* object based on the original,
        optionally modifying the number of states carried in the array.

        :param nstates: Optional parameter specifying the number
              of states (or bands) to be carried in the array.
              Defaults to the same as the original *WFArray* object.

        :returns:
          * **wf_new** -- returns a similar WFArray except that array
              elements are unitialized and the number of states may have
              changed.

        Example usage::

          # Make new empty WFArray object containing 6 bands per k-point
          wf_new=wf.empty_like(nstates=6)

        """

        # make a full copy of the WFArray
        wf_new = copy.deepcopy(self)

        if nstates is None:
            wf_new._wfs = np.empty_like(wf_new._wfs)
        else:
            wf_shape = list(wf_new._wfs.shape)
            # modify numer of states (after k indices & before orb and spin)
            wf_shape[self._dim_mesh] = nstates
            wf_new._wfs = np.empty_like(wf_new._wfs, shape=wf_shape)

        return wf_new
    
    
    def _apply_phase(self, inverse=False):
        """
        Change between cell periodic and Bloch wfs by multiplying exp(\pm i k . tau)

        Returns:
            wfsxphase (np.ndarray): wfs with orbitals multiplied by phase factor
        """
        lam = -1 if inverse else 1  # overall minus if getting cell periodic from Bloch

        # create a regular mesh of k-points
        # NOTE: Assumes that the WFArray was populated using a regular mesh
        # of k-points

        nks = self.mesh_size
        dim_k = len(nks)
        if dim_k != self.model.dim_k:
            raise ValueError(
                f"WFArray has {dim_k} k-dimensions, but model has {self.model.dim_k}!"
            )
        # create a mesh of k-points in the range [0, 1) for each k-dimension
        end_pts = [0, 1]
        k_vals = [
            np.linspace(end_pts[0], end_pts[1], nk, endpoint=False)
            for nk in nks
        ]
        flat_mesh = np.stack(np.meshgrid(*k_vals, indexing="ij"), axis=-1)
        flat_mesh = flat_mesh.reshape(-1, dim_k)
        # flat_mesh is now of shape [k_val, dim_k], where k_val is the total number of k-points

        per_dir = list(
            range(flat_mesh.shape[-1])
        ) 
        # slice second dimension to only keep only periodic dimensions in orb
        per_orb = self.model.orb_vecs[:, per_dir]

        # compute a list of phase factors: exp(pm i k . tau) of shape [k_val, orbital]
        phases = np.exp(
            lam * 1j * 2 * np.pi * per_orb @ flat_mesh.T, dtype=complex
        ).T
        # reshape phases to have shape: [nk1, nk2, ..., nkd, norb]
        phases = phases.reshape(*nks, self.norb)

        # wfs is of shape [nk1, nk2, ..., nkd, nband, norb, nspin]
        wfs = self.wfs

        # broadcasting to match dimensions of wfs
        if self._nspin == 1:
            # newaxis along state dimension
            phases = phases[..., np.newaxis, :]
        elif self._nspin == 2:
            # newaxis along state and spin dimension
            phases = phases[..., np.newaxis, :, np.newaxis]

        return wfs * phases


    def impose_pbc(self, mesh_dir: int, k_dir: int):
        r"""

        If the *WFArray* object was populated using the
        :func:`pythtb.WFArray.solve_on_grid` method, this function
        should not be used since it will be called automatically by
        the code.

        The eigenfunctions :math:`\Psi_{n {\bf k}}` are by convention
        chosen to obey a periodic gauge, i.e.,
        :math:`\Psi_{n,{\bf k+G}}=\Psi_{n {\bf k}}` not only up to a
        phase, but they are also equal in phase.  It follows that
        the cell-periodic Bloch functions are related by
        :math:`u_{n,{\bf k+G}}=e^{-i{\bf G}\cdot{\bf r}} u_{n {\bf k}}`.
        See :download:`notes on tight-binding formalism
        <misc/pythtb-formalism.pdf>` section 4.4 and equation 4.18 for
        more detail.  This routine sets the cell-periodic Bloch function
        at the end of the string in direction :math:`{\bf G}` according
        to this formula, overwriting the previous value.

        This function will impose these periodic boundary conditions along
        one direction of the array. We are assuming that the k-point
        mesh increases by exactly one reciprocal lattice vector along
        this direction. This is currently **not** checked by the code;
        it is the responsibility of the user. Currently *WFArray*
        does not store the k-vectors on which the model was solved;
        it only stores the eigenvectors (wavefunctions).

        :param mesh_dir: Direction of WFArray along which you wish to
          impose periodic boundary conditions.

        :param k_dir: Corresponding to the periodic k-vector direction
          in the Brillouin zone of the underlying *TBModel*.  Since
          version 1.7.0 this parameter is defined so that it is
          specified between 0 and *dim_r-1*.

        See example :ref:`3site_cycle-example`, where the periodic boundary
        condition is applied only along one direction of *WFArray*.

        Example usage::

          # Imposes periodic boundary conditions along the mesh_dir=0
          # direction of the WFArray object, assuming that along that
          # direction the k_dir=1 component of the k-vector is increased
          # by one reciprocal lattice vector.  This could happen, for
          # example, if the underlying TBModel is two dimensional but
          # WFArray is a one-dimensional path along k_y direction.
          wf.impose_pbc(mesh_dir=0,k_dir=1)

        """

        if k_dir not in self._model._per:
            raise Exception(
                "Periodic boundary condition can be specified only along periodic directions!"
            )
        
        if not _is_int(mesh_dir):
            raise TypeError("mesh_dir should be an integer!")
        if mesh_dir < 0 or mesh_dir >= self.dim_mesh:
            raise IndexError("mesh_dir outside the range!")
        
        self._pbc_axes.append(mesh_dir)

        # Compute phase factors from orbital vectors dotted with G parallel to k_dir
        ffac = np.exp(-2j * np.pi * self._orb[:, k_dir])
        if self.nspin == 1:
            phase = ffac
        else:
            # for spinors, same phase multiplies both components
            phase = np.zeros((self.norb, 2), dtype=complex)
            phase[:, 0] = ffac
            phase[:, 1] = ffac


        # mesh_dir is the direction of the array along which we impose pbc
        # and it is also the direction of the k-vector along which we
        # impose pbc e.g.
        # mesh_dir=0 corresponds to kx, mesh_dir=1 to ky, etc.
        # mesh_dir=2 corresponds to lambda, etc.

        ### Define slices in a way that is general for arbitrary dimensions ###
        # Example: mesh_dir = 2 (2 defines the axis in Python counting)
        # add one for Python counting and one for ellipses 
        slc_lft = [slice(None)]*(mesh_dir+2) # e.g., [:, :, :, :]
        slc_rt = [slice(None)]*(mesh_dir+2) # e.g., [:, :, :, :]
        # last element along mesh_dir axis
        slc_lft[mesh_dir] = -1 # e.g., [:, :, -1, :]
        # first element along mesh_dir axis
        slc_rt[mesh_dir] = 0 # e.g., [:, :, 0, :]
        # take all components of remaining axes with ellipses
        slc_lft[mesh_dir+1] = Ellipsis # e.g., [:, :, -1, ...]
        slc_rt[mesh_dir+1] = Ellipsis # e.g., [:, :, 0, ...]

        # Set the last point along mesh_dir axis equal to first 
        # multiplied by the phase factor
        self._wfs[tuple(slc_lft)] = self._wfs[tuple(slc_rt)] * phase


    def impose_loop(self, mesh_dir):
        r"""

        If the user knows that the first and last points along the
        *mesh_dir* direction correspond to the same Hamiltonian (this
        is **not** checked), then this routine can be used to set the
        eigenvectors equal (with equal phase), by replacing the last
        eigenvector with the first one (for each band, and for each
        other mesh direction, if any).

        This routine should not be used if the first and last points
        are related by a reciprocal lattice vector; in that case,
        :func:`pythtb.WFArray.impose_pbc` should be used instead.

        :param mesh_dir: Direction of WFArray along which you wish to
          impose periodic boundary conditions.

        Example usage::

          # Suppose the WFArray object is three-dimensional
          # corresponding to (kx,ky,lambda) where (kx,ky) are
          # wavevectors of a 2D insulator and lambda is an
          # adiabatic parameter that goes around a closed loop.
          # Then to insure that the states at the ends of the lambda
          # path are equal (with equal phase) in preparation for
          # computing Berry phases in lambda for given (kx,ky),
          # do wf.impose_loop(mesh_dir=2)

        """
        if not _is_int(mesh_dir):
            raise TypeError("mesh_dir must be an integer.")
        if mesh_dir < 0 or mesh_dir >= self.dim_mesh:
            raise ValueError(
                f"mesh_dir must be between 0 and {self.dim_mesh-1}, got {mesh_dir}."
            )
        
        self._loop_axes.append(mesh_dir)

        slc_lft = [slice(None)]*(mesh_dir+2) # e.g., [:, :, :, :]
        slc_rt = [slice(None)]*(mesh_dir+2) # e.g., [:, :, :, :]

        slc_lft[mesh_dir] = -1 # e.g., [:, :, -1, :]
        slc_rt[mesh_dir] = 0 # e.g., [:, :, 0, :]
        slc_lft[mesh_dir+1] = Ellipsis # e.g., [:, :, -1, ...]
        slc_rt[mesh_dir+1] = Ellipsis # e.g., [:, :, 0, ...]
        # set the last point in the mesh_dir direction equal to the first one
        self._wfs[tuple(slc_lft)] = self._wfs[tuple(slc_rt)]


    def position_matrix(self, key, occ, dir):
        """Similar to :func:`pythtb.TBModel.position_matrix`.  Only
        difference is that, in addition to specifying *dir*, one also
        has to specify *key* (k-point of interest) and *occ* (list of
        states to be included, which can optionally be 'All')."""

        # Check for special case of parameter occ
        if isinstance(occ, str) and occ.lower() == "all":
            occ = np.arange(self.nstates, dtype=int)
        elif isinstance(occ, (list, np.ndarray, tuple, range)):
            occ = list(occ)
            occ = np.array(occ, dtype=int)
        else:
            raise TypeError(
            "occ must be a list, numpy array, tuple, or 'all' defining "
            "band indices of itnterest."
            )

        if occ.ndim != 1:
            raise Exception(
                """\n\nParameter occ must be a one-dimensional array or string "All"."""
            )

        # check if model came from w90
        if not self._model._assume_position_operator_diagonal:
            _offdiag_approximation_warning_and_stop()
        #
        evec = self._wfs[tuple(key)][occ]
        return self._model.position_matrix(evec, dir)


    def position_expectation(self, key, occ, dir):
        """Similar to :func:`pythtb.TBModel.position_expectation`.  Only
        difference is that, in addition to specifying *dir*, one also
        has to specify *key* (k-point of interest) and *occ* (list of
        states to be included, which can optionally be 'All')."""

        # Check for special case of parameter occ
        if isinstance(occ, str) and occ.lower() == "all":
            occ = np.arange(self.nstates, dtype=int)
        elif isinstance(occ, (list, np.ndarray, tuple, range)):
            occ = list(occ)
            occ = np.array(occ, dtype=int)
        else:
            raise TypeError(
            "occ must be a list, numpy array, tuple, or 'all' defining "
            "band indices of itnterest."
            )

        if occ.ndim != 1:
            raise Exception(
                """\n\nParameter occ must be a one-dimensional array or string "all"."""
            )

        # check if model came from w90
        if not self.model._assume_position_operator_diagonal:
            _offdiag_approximation_warning_and_stop()
    
        evec = self.wfs[tuple(key)][occ]
        return self.model.position_expectation(evec, dir)


    def position_hwf(self, key, occ, dir, hwf_evec=False, basis="wavefunction"):
        """Similar to :func:`pythtb.TBModel.position_hwf`, except that
        in addition to specifying *dir*, one also has to specify
        *key*, the k-point of interest, and *occ*, a list of states to
        be included (typically the occupied states).

        For backwards compatibility the default value of *basis* here is different
        from that in :func:`pythtb.TBModel.position_hwf`.
        """
        # Check for special case of parameter occ
        if isinstance(occ, str) and occ.lower() == "all":
            occ = np.arange(self.nstates, dtype=int)
        elif isinstance(occ, (list, np.ndarray, tuple, range)):
            occ = list(occ)
            occ = np.array(occ, dtype=int)
        else:
            raise TypeError(
            "occ must be a list, numpy array, tuple, or 'all' defining "
            "band indices of itnterest."
            )
        if occ.ndim != 1:
            raise Exception(
                """\n\nParameter occ must be a one-dimensional array or string "all"."""
            )

        # check if model came from w90
        if not self.model._assume_position_operator_diagonal:
            _offdiag_approximation_warning_and_stop()

        evec = self.wfs[tuple(key)][occ]
        return self.model.position_hwf(evec, dir, hwf_evec, basis)
    
    def get_links(self, state_idx=None, dirs=None):
        """
        Compute the links (unitary matrices) for the wavefunctions
        in the *WFArray* object along a given direction.
        The links are computed as the unitary part of the overlap between the 
        wavefunctions at neighboring points in each mesh direction.

        The links are computed for all states in the *WFArray* object,
        and the resulting unitary matrices are returned in an array
        of shape [dim, nk1, nk2, ..., nkd, n_states, n_states],
        where dim is the number of dimensions of the mesh, nk1, nk2, ..., nkd
        are the sizes of the mesh in each dimension, and n_states is the number of states
        in the *WFArray* object.

        Args:
            state_idx (int or list of int):
                Index or indices of the states for which to compute the links.
                If an integer is provided, only that state will be considered.
                If a list is provided, links for all specified states will be computed.
            dirs (list of int, optional):
                List of directions along which to compute the links.
                If not provided, links will be computed for all directions in the mesh.
        Returns:
            U_forward (np.ndarray):
                Array of shape [dim, nk1, nk2, ..., nkd, n_states, n_states]
                containing the unitary matrices for the forward links in each direction.
        """
        wfs = self.get_states(flatten_spin=True)

        # State selection
        if state_idx is not None:
            if isinstance(state_idx, (list, np.ndarray)):
                # If state_idx is a list or array, select those states
                state_idx = np.array(state_idx, dtype=int)
            elif isinstance(state_idx, int):
                # If state_idx is a single integer, convert to array
                state_idx = np.array([state_idx], dtype=int)
            else:
                raise TypeError("state_idx must be an integer, list, or numpy array.")
            
            wfs = np.take(wfs, state_idx, axis=-2)

        if dirs is None:
            # If no specific directions are provided, compute links for all directions
            dirs = list(range(self.dim_mesh))

        # n_states = wfs.shape[-2]
        # n_param = self.mesh_size
        # wfs_flat = wfs.reshape(*n_param, n_states, -1)

        U_forward = []
        for mu in dirs:
            print(f"Computing links for direction mu={mu}")
            wfs_shifted = np.roll(wfs, -1, axis=mu)
            
            # <u_nk| u_m k+delta_mu>
            ovr_mu = wfs.conj() @ wfs_shifted.swapaxes(-2, -1)

            U_forward_mu = np.zeros_like(ovr_mu, dtype=complex)
            V, _, Wh = np.linalg.svd(ovr_mu, full_matrices=False)
            U_forward_mu = V @ Wh
            U_forward.append(U_forward_mu)

        return np.array(U_forward)
    
    @staticmethod
    def wilson_loop(wfs_loop, evals=False):
        """Compute Wilson loop unitary matrix and its eigenvalues for multiband Berry phases.

        Multiband Berry phases always returns numbers between -pi and pi.

        Args:
            wfs_loop (np.ndarray):
                Has format [loop_idx, band, orbital(, spin)] and loop has to be one dimensional.
                Assumes that first and last loop-point are the same. Therefore if
                there are n wavefunctions in total, will calculate phase along n-1
                links only!
            berry_evals (bool):
                If berry_evals is True then will compute phases for
                individual states, these corresponds to 1d hybrid Wannier
                function centers. Otherwise just return one number, Berry phase.
        """
        # check that wfs_loop has appropriate shape
        if wfs_loop.ndim < 3 or wfs_loop.ndim > 4:
            raise ValueError(
                "wfs_loop must be a 3D or 4D array with shape [loop_idx, band, orbital(, spin)]"
            )

        # check if there is a spin axis, then flatten
        is_spin = wfs_loop.ndim == 4 and wfs_loop.shape[-1] == 2
        if is_spin:
            # flatten spin axis
            wfs_loop = wfs_loop.reshape(wfs_loop.shape[0], wfs_loop.shape[1], -1, 2)

        ovr_mats = wfs_loop[:-1].conj() @ wfs_loop[1:].swapaxes(-2, -1)
        V, _, Wh = np.linalg.svd(ovr_mats, full_matrices=False)
        U_link = V @ Wh
        U_wilson = U_link[0]
        for i in range(1, len(U_link)):
            U_wilson = U_wilson @ U_link[i]

        # calculate phases of all eigenvalues
        if evals:
            evals = np.linalg.eigvals(U_wilson)  # Wilson loop eigenvalues
            eval_pha = -np.angle(evals)  # Multiband  Berrry phases
            return U_wilson, eval_pha
        else:
            return U_wilson
        
    @staticmethod
    def berry_loop(wfs_loop, evals=False):
        r"""
        Computes the Berry phase along a one-dimensional loop of
        wavefunctions. The loop is assumed to be one-dimensional,
        meaning that the first and last points in the loop are
        assumed to be the same, and the wavefunctions at these
        points are also assumed to be the same.

        The wavefunctions in the loop should be ordered such that
        the first point corresponds to the first wavefunction, the
        second point to the second wavefunction, and so on, up to
        the last point, which corresponds to the last wavefunction.
        The wavefunctions should be in the format [loop_idx, band, orbital, spin],
        where loop_idx is the index of the wavefunction in the loop.

        The Berry phase is computed as the logarithm of the determinant
        of the product of the overlap matrices between neighboring
        wavefunctions in the loop. The Berry phase is returned as a
        single number, which is the total Berry phase for the loop.

        Args:
            wfs_loop (np.ndarray): Wavefunctions in the loop, with shape
                [loop_idx, band, orbital, spin]. The first and last points
                in the loop are assumed to be the same.
            evals (bool): If True, will return the eigenvalues of the Wilson loop
                unitary matrix instead of the Berry phase. The eigenvalues
                correspond to the "maximally localized Wannier centers" or
                "Wilson loop eigenvalues". If False, will return the total
                Berry phase for the loop.
        Returns:
            np.ndarray: If evals is True, returns the eigenvalues of the Wilson loop
                unitary matrix, which are the Berry phases for each band.
                If evals is False, returns the total Berry phase for the loop,
                which is a single number.
        """

        U_wilson = WFArray.wilson_loop(wfs_loop, evals=evals)

        if evals:
            hwf_centers = U_wilson[1]
            return hwf_centers
        else:
            berry_phase = -np.angle(np.linalg.det(U_wilson))       
            return berry_phase

    def berry_phase(self, occ="All", dir=None, contin=True, berry_evals=False):
        r"""

        Computes the Berry phase along a given array direction
        and for a given set of states.  These are typically the
        occupied Bloch states, in which case *occ* should range
        over all occupied bands.  In this context, the occupied
        and unoccupied bands should be well separated in energy;
        it is the responsibility of the user to check that this
        is satisfied.  If *occ* is not specified or is specified
        as 'All', all states are selected. By default, the
        function returns the Berry phase traced over the
        specified set of bands, but optionally the individual
        phases of the eigenvalues of the global unitary rotation
        matrix (corresponding to "maximally localized Wannier
        centers" or "Wilson loop eigenvalues") can be requested
        (see parameter *berry_evals* for more details).

        For an array of size *N* in direction $dir$, the Berry phase
        is computed from the *N-1* inner products of neighboring
        eigenfunctions.  This corresponds to an "open-path Berry
        phase" if the first and last points have no special
        relation.  If they correspond to the same physical
        Hamiltonian, and have been properly aligned in phase using
        :func:`pythtb.WFArray.impose_pbc` or
        :func:`pythtb.WFArray.impose_loop`, then a closed-path
        Berry phase will be computed.

        For a one-dimensional WFArray (i.e., a single string), the
        computed Berry phases are always chosen to be between -pi and pi.
        For a higher dimensional WFArray, the Berry phase is computed
        for each one-dimensional string of points, and an array of
        Berry phases is returned. The Berry phase for the first string
        (with lowest index) is always constrained to be between -pi and
        pi. The range of the remaining phases depends on the value of
        the input parameter *contin*.

        The discretized formula used to compute Berry phase is described
        in Sec. 4.5 of :download:`notes on tight-binding formalism
        <misc/pythtb-formalism.pdf>`.

        :param occ: Optional array of indices of states to be included
          in the subsequent calculations, typically the indices of
          bands considered occupied.  Default is all bands.

        :param dir: Index of WFArray direction along which Berry phase is
          computed. This parameters needs not be specified for
          a one-dimensional WFArray.

        :param contin: Optional boolean parameter. If True then the
          branch choice of the Berry phase (which is indeterminate
          modulo 2*pi) is made so that neighboring strings (in the
          direction of increasing index value) have as close as
          possible phases. The phase of the first string (with lowest
          index) is always constrained to be between -pi and pi. If
          False, the Berry phase for every string is constrained to be
          between -pi and pi. The default value is True.

        :param berry_evals: Optional boolean parameter. If True then
          will compute and return the phases of the eigenvalues of the
          product of overlap matrices. (These numbers correspond also
          to hybrid Wannier function centers.) These phases are either
          forced to be between -pi and pi (if *contin* is *False*) or
          they are made to be continuous (if *contin* is True).

        :returns:
          * **pha** -- If *berry_evals* is False (default value) then
            returns the Berry phase for each string. For a
            one-dimensional WFArray this is just one number. For a
            higher-dimensional WFArray *pha* contains one phase for
            each one-dimensional string in the following format. For
            example, if *WFArray* contains k-points on mesh with
            indices [i,j,k] and if direction along which Berry phase
            is computed is *dir=1* then *pha* will be two dimensional
            array with indices [i,k], since Berry phase is computed
            along second direction. If *berry_evals* is True then for
            each string returns phases of all eigenvalues of the
            product of overlap matrices. In the convention used for
            previous example, *pha* in this case would have indices
            [i,k,n] where *n* refers to index of individual phase of
            the product matrix eigenvalue.

        Example usage::

          # Computes Berry phases along second direction for three lowest
          # occupied states. For example, if wf is threedimensional, then
          # pha[2,3] would correspond to Berry phase of string of states
          # along wf[2,:,3]
          pha = wf.berry_phase([0, 1, 2], 1)

        See also these examples: :ref:`haldane_bp-example`,
        :ref:`cone-example`, :ref:`3site_cycle-example`,

        """
        # Get wavefunctions in the array, flattening spin if necessary
        # wfs is of shape [nk1, nk2, ..., nkd, nstate, nstate]
        wfs = self.get_states(flatten_spin=True)

        # Check for special case of parameter occ
        if isinstance(occ, str) and occ.lower() == "all":
            occ = np.arange(self.nstates, dtype=int)
        elif isinstance(occ, (list, np.ndarray, tuple, range)):
            occ = np.array(list(occ), dtype=int)
        else:
            raise TypeError(
            "occ must be a list, numpy array, tuple, or 'all' defining "
            "band indices of itnterest."
            )
        
        if occ.ndim != 1:
            raise ValueError(
                """Parameter occ must be a one-dimensional array or "all"."""
            )

        # check if model came from w90
        if not self.model._assume_position_operator_diagonal:
            _offdiag_approximation_warning_and_stop()


        # number of mesh dimensions is total dims minus band and orbital axes
        mesh_axes = wfs.ndim - 2
        # Validate dir parameter
        if dir is None:
            if mesh_axes != 1:
                raise ValueError("If dir is not specified, the mesh must be one-dimensional.")
            dir = 0
        if dir is not None and (dir < 0 or dir >= mesh_axes):
            raise ValueError("dir must be between 0 and number of mesh dimensions - 1")
        
        # Prepare wavefunctions: select occupied bands and bring loop dimension first
        wf = wfs[..., occ, :]
        wf = np.moveaxis(wf, dir, 0)  # shape: (N_loop, *rest, nbands)
        N_loop, *rest_shape, nbands, norb = wf.shape
        # Flatten redundant param dimensions intermediately
        wf_flat = wf.reshape(N_loop, -1, nbands, norb)  # shape: (N_loop, rest_shape, nbands, norb)

        # Compute Berry phase for each slice along other dimensions
        results = []
        # loop over all other parameter values other than the loop dimension
        for idx in range(wf_flat.shape[1]):
            slice_wf = wf_flat[:, idx, :, :]
            results.append(self.berry_loop(slice_wf, evals=berry_evals))

        ret = np.array(results)

        if contin:
            # Make phases continuous 
            ret = np.unwrap(ret, axis=0)
                
        return ret

    def berry_flux(self, state_idx=None, plane=None, abelian=True):
        r"""

        In the case of a 2-dimensional *WFArray* array calculates the
        integral of Berry curvature over the entire plane.  In higher
        dimensional case (3 or 4) it will compute integrated curvature
        over all 2-dimensional slices of a higher-dimensional
        *WFArray*.

        :param state_idx: Optional array of indices of states to be included
          in the subsequent calculations, typically the indices of
          bands considered occupied. If not specified, or None, all bands are
          included.

        :param plane: Array or tuple of two indices defining the axes in the
            WFArray mesh which the Berry flux is computed over. By default, 
            all directions are considered, and the full Berry flux tensor is
            returned.

        :param abelian: If *True* then the Berry flux is computed
          using the abelian formula, which corresponds to the band-traced
          non-Abelian Berry curvature. If *False* then the non-Abelian Berry
          flux tensor is computed. Default value is *True*.

        :param integrate: If *True* then the plaquette fluxes are summed to 
          return the integrated Berry flux over the entire plane.
          If *False* then the function returns the Berry phase around each
          plaquette in the array. In the 2-dimensional case this
          corresponds to the integral of Berry curvature over the entire
          plane, while in higher dimensions it corresponds to the integral of
          Berry curvature over all slices defined with directions *dirs*.

        :returns:

          * **flux** -- 
            The Berry flux tensor, which is an array of general shape
            [dim_mesh, dim_mesh, *flux_shape, n_states, n_states]. The 
            shape will depend on the parameters passed to the function.

            If plane is *None* (default), then the first two axes 
            (dim_mesh, dim_mesh) correspond to the plane directions, otherwise, 
            these axes are absent. 

            If *abelian* is *False* then the last two axes are the band indices
            running over the selected *state_idx* indices.
            If *abelian* is *True* (default) then the last two axes are absent, and
            the returned flux is a scalar value, not a matrix. 

        Example usage::

          # Computes Berry curvature of first three bands in 2D model
          flux = wf.berry_flux([0, 1, 2]) # shape: (dim1, dim2, nk1, nk2)
          flux = wf.berry_flux([0, 1, 2], plane=(0, 1)) # shape: (nk1, nk2)
          flux = wf.berry_flux([0, 1, 2], plane=(0, 1), abelian=False) # shape: (nk1, nk2, n_states, n_states)

          # 3D model example
          flux = wf.berry_flux([0, 1, 2], plane=(0, 1)) # shape: (nk1, nk2, nk3)
        """
        # Validate state_idx
        if state_idx is None:
            state_idx = np.arange(self.nstates)
        elif isinstance(state_idx, (list, np.ndarray, tuple)):
            state_idx = np.array(state_idx, dtype=int)
            if state_idx.ndim != 1:
                raise ValueError("state_idx must be a one-dimensional array.")
            if np.any(state_idx < 0) or np.any(state_idx >= self.nstates):
                raise ValueError(f"state_idx must be between 0 and {self.nstates-1}.")
        else:
            raise TypeError("state_idx must be None, a list, tuple, or numpy array.")
        if len(state_idx) == 0:
            raise ValueError("state_idx cannot be empty.")
        if np.any(np.diff(state_idx) < 0):
            raise ValueError("state_idx must be sorted in ascending order.")
        

        n_states = len(state_idx) # Number of states considered
        dim_mesh = self.dim_mesh  # Total dimensionality of adiabatic space: d
        n_param = list(self.mesh_size)  # Number of points in adiabatic mesh: (nk1, nk2, ..., nkd)

        # Validate plane
        if plane is not None and not isinstance(plane, (list, tuple, np.ndarray)):
            raise TypeError("plane must be None, a list, tuple, or numpy array.")
        if len(plane) != 2:
            raise ValueError("plane must contain exactly two directions.")
        if any(p < 0 or p >= dim_mesh for p in plane):
            raise ValueError(f"Plane indices must be between 0 and {dim_mesh-1}.")
        if plane[0] == plane[1]:
            raise ValueError("Plane indices must be different.")

        # Unique axes for periodic boundary conditions and loops
        pbc_axes = list(set(self._pbc_axes + self._loop_axes))
        flux_shape = n_param
        for ax in pbc_axes:
            flux_shape[ax] -= 1  # Remove last link in each periodic direction

        # Initialize the Berry flux array
        if plane is None:
            shape = (
                (dim_mesh, dim_mesh, *flux_shape, n_states, n_states)
                if not abelian
                else (dim_mesh, dim_mesh, *flux_shape)
            )
            berry_flux = np.zeros(shape, dtype=complex)
            dirs = list(range(dim_mesh))
            plane_idxs = dim_mesh
        else:
            p, q = plane  # Unpack plane directions

            shape = (
                (*flux_shape, n_states, n_states)
                if not abelian
                else (*flux_shape,)
            )
            berry_flux = np.zeros(shape, dtype=float)

            dirs = [p, q]
            plane_idxs = 2

        # U_forward: Overlaps <u_{nk} | u_{n, k+delta k_mu}>
        U_forward = self.get_links(state_idx=state_idx, dirs=dirs)

        # remove last links in mesh if pbc or loop is imposed along plane directions
        for ax in pbc_axes:
            # remove ax+1 (+1 skips first axis, which is the loop direction)
            U_forward = np.delete(U_forward, -1, axis=ax+1)

        # Compute Berry flux for each pair of states
        for mu in range(plane_idxs):
            for nu in range(mu + 1, plane_idxs):
                print(f"Computing flux in plane: mu={mu}, nu={nu}")
                U_mu = U_forward[mu]
                U_nu = U_forward[nu]

                # Shift the wavefunctions along the mu and nu directions
                U_nu_shift_mu = np.roll(U_nu, -1, axis=mu)
                U_mu_shift_nu = np.roll(U_mu, -1, axis=nu)

                # Wilson loops: W = U_{mu}(k_0) U_{nu}(k_0+delta_mu) U^{-1}_{mu}(k_0+delta_mu+delta_nu) U^{-1}_{nu}(k_0)
                U_wilson = np.matmul(
                    np.matmul(
                        np.matmul(U_mu, U_nu_shift_mu),
                        U_mu_shift_nu.conj().swapaxes(-1, -2),
                    ),
                    U_nu.conj().swapaxes(-1, -2),
                )

                if not abelian:
                    eigvals, eigvecs = np.linalg.eig(U_wilson)
                    angles = -np.angle(eigvals)
                    angles_diag = np.einsum(
                        "...i, ij -> ...ij", angles, np.eye(angles.shape[-1])
                    )
                    eigvecs_inv = np.linalg.inv(eigvecs)
                    phases_plane = np.matmul(
                        np.matmul(eigvecs, angles_diag), eigvecs_inv
                    )
                else:
                    det_U = np.linalg.det(U_wilson)
                    phases_plane = -np.angle(det_U)

                if plane is None:
                    # Store the Berry flux in a 2D array for each pair of directions
                    berry_flux[mu, nu] = phases_plane
                    berry_flux[nu, mu] = -phases_plane
                else:   
                    berry_flux = phases_plane.real
       
        return berry_flux
    
    
    def chern_num(self, plane=(0, 1), state_idx=None):
        r"""
        Computes the Chern number for a *WFArray* in the specified plane.
        The Chern number is computed as the integral of the Berry flux
        over the specified plane, divided by 2 * pi.
        The plane is specified by a tuple of two indices, which correspond
        to the directions in the parameter mesh. 

        :param plane: Tuple of two indices specifying the plane in which
            the Chern number is computed. The indices should be between 0
            and the number of mesh dimensions minus 1. If None, the
            Chern number is computed for the first two dimensions of the mesh.

        :param state_idx: Optional array of indices of states to be included
          in the Chern number calculation. If None, all states are included.

        :returns: The Chern number for the specified plane. If the WFArray
            is defined in a higher-dimensional space, the Chern number
            is computed for each 2D slice of the higher-dimensional space. 
            The shape of the returned array is (nk3, ..., nkd) if the plane is (0, 1),
            where nk3, ..., nkd are the sizes of the mesh in the remaining dimensions.

        Example usage::
            chern = wfs.chern_num(plane=(0, 1), state_idx=np.arange(n_occ))
            # shape: (nk3, nk4, ..., nkd)

        """
        if state_idx is None:
            state_idx = np.arange(self.nstates)  # assume half-filled occupied

        # shape of the Berry flux array: (nk1, nk2, ..., nkd)
        berry_flux = self.berry_flux(state_idx=state_idx, plane=plane, abelian=True)
        # shape of chern (if plane is (0,1)): (nk3, ..., nkd) 
        chern = np.sum(berry_flux, axis=plane) / (2 * np.pi)

        return chern


class Bloch(WFArray):
    def __init__(self, model: TBModel, *param_dims):
        """Class for storing and manipulating Bloch like wavefunctions.

        Wavefunctions are defined on a semi-full reciprocal space mesh.
        """
        super().__init__(model, param_dims)
        assert (
            len(param_dims) >= model.dim_k
        ), "Number of dimensions must be >= number of reciprocal space dimensions"

        self.model: TBModel = model
        # model attributes
        self._n_orb = model.get_num_orbitals()
        self._nspin = self.model.nspin
        self._n_states = self._n_orb * self._nspin

        # reciprocal space dimensions
        self.dim_k = model.dim_k
        self.nks = param_dims[: self.dim_k]
        # set k_mesh
        self.model.set_k_mesh(*self.nks)
        # stores k-points on a uniform mesh, calculates nearest neighbor points given the model lattice
        self.k_mesh: KMesh = model.k_mesh

        # adiabatic dimension
        self.dim_lam = len(param_dims) - self.dim_k
        self.n_lambda = param_dims[self.dim_k :]

        # Total adiabatic parameter space
        self.dim_param = self.dim_adia = self.dim_k + self.dim_lam
        self.n_param = self.n_adia = (*self.nks, *self.n_lambda)

        # periodic boundary conditions assumed True unless specified
        self.pbc_lam = True

        # axes indexes
        self.k_axes = tuple(range(self.dim_k))
        self.lambda_axes = tuple(range(self.dim_k, self.dim_param))

        if self._nspin == 2:
            self.spin_axis = -1
            self.orb_axis = -2
            self.state_axis = -3
        else:
            self.spin_axis = None
            self.orb_axis = -1
            self.state_axis = -2

        # wavefunction shapes
        if self.dim_lam > 0:
            if self._nspin == 2:
                self._wf_shape = (
                    *self.nks,
                    *self.n_lambda,
                    self._n_states,
                    self._n_orb,
                    self._nspin,
                )
            else:
                self._wf_shape = (
                    *self.nks,
                    *self.n_lambda,
                    self._n_states,
                    self._n_orb,
                )
        else:
            if self._nspin == 2:
                self._wf_shape = (*self.nks, self._n_states, self._n_orb, self._nspin)
            else:
                self._wf_shape = (*self.nks, self._n_states, self._n_orb)

        # self.set_Bloch_ham()
    @property
    def u_wfs(self):
        return self._u_wfs
    
    @property
    def psi_wfs(self):
        return self._psi_wfs


    def get_wf_axes(self):
        dict_axes = {
            "wf shape": self._wf_shape,
            "Number of axes": len(self._wf_shape),
            "k-axes": self.k_axes,
            "lambda-axes": self.lambda_axes,
            "spin-axis": self.spin_axis,
            "orbital axis": self.orb_axis,
            "state axis": self.state_axis,
        }
        return dict_axes

    def set_pbc_lam(self):
        self.pbc_lam = True

    def set_Bloch_ham(self, lambda_vals=None, model_fxn=None):
        if lambda_vals is None:
            H_k = self.model.get_ham(k_pts=self.k_mesh.flat_mesh)  # [Nk, norb, norb]
            # [nk1, nk2, ..., norb, norb]
            self.H_k = H_k.reshape(*[nk for nk in self.k_mesh.nks], *H_k.shape[1:])
            return

        lambda_keys = list(lambda_vals.keys())
        lambda_ranges = list(lambda_vals.values())
        lambda_shape = tuple(len(vals) for vals in lambda_ranges)
        dim_lambda = len(lambda_keys)

        n_kpts = self.k_mesh.Nk
        n_orb = self._n_orb
        n_spin = self._n_spin
        n_states = n_orb * n_spin

        # Initialize storage for wavefunctions and energies
        if n_spin == 1:
            H_kl = np.zeros((*lambda_shape, n_kpts, n_states, n_states), dtype=complex)
        elif n_spin == 2:
            H_kl = np.zeros(
                (*lambda_shape, n_kpts, n_orb, n_spin, n_orb, n_spin), dtype=complex
            )

        for idx, param_set in enumerate(np.ndindex(*lambda_shape)):
            # kwargs for model_fxn with specified parameter values
            param_dict = {
                lambda_keys[i]: lambda_ranges[i][param_set] for i in range(dim_lambda)
            }

            # Generate the model with modified parameters
            modified_model = model_fxn(**param_dict)

            H_kl[param_set] = modified_model.get_ham(k_pts=self.k_mesh.flat_mesh)

        # Reshape for compatibility with existing Berry curvature methods

        if self._nspin == 1:
            new_axes = (
                (dim_lambda,)
                + tuple(range(dim_lambda))
                + tuple(range(dim_lambda + 1, dim_lambda + 3))
            )
        else:
            new_axes = (
                (dim_lambda,)
                + tuple(range(dim_lambda))
                + tuple(range(dim_lambda + 1, dim_lambda + 5))
            )
        H_kl = np.transpose(H_kl, axes=new_axes)

        if self._nspin == 1:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_states)
        else:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_orb, n_spin)
        H_kl = H_kl.reshape(new_shape)

        self.H_k = H_kl

    def solve_model(self, model_fxn=None, lambda_vals=None):
        """
        Solves for the eigenstates of the Bloch Hamiltonian defined by the model over a semi-full
        k-mesh, e.g. in 3D reduced coordinates {k = [kx, ky, kz] | k_i in [0, 1)}.

        Args:
            model_fxn (function, optional):
                A function that returns a model given a set of parameters.
            param_vals (dict, optional):
                Dictionary of parameter values for adiabatic evoltuion. Each key corresponds to
                a varying parameter and the values are arrays
        """

        if lambda_vals is None:
            # compute eigenstates and eigenenergies on full k_mesh
            eigvals, eigvecs = self.model.solve_ham(
                self.k_mesh.flat_mesh, return_eigvecs=True
            )
            eigvecs = eigvecs.reshape(*self.k_mesh.nks, *eigvecs.shape[1:])
            eigvals = eigvals.reshape(*self.k_mesh.nks, *eigvals.shape[1:])
            self.set_wfs(eigvecs)
            self.energies = eigvals
            self.is_energy_eigstate = True
            return

        lambda_keys = list(lambda_vals.keys())
        lambda_ranges = list(lambda_vals.values())
        lambda_shape = tuple(len(vals) for vals in lambda_ranges)
        dim_lambda = len(lambda_keys)

        n_kpts = self.k_mesh.Nk
        n_orb = self.model.norb
        n_spin = self.model.nspin
        n_states = n_orb * n_spin

        # Initialize storage for wavefunctions and energies
        if n_spin == 1:
            u_wfs = np.zeros((*lambda_shape, n_kpts, n_states, n_states), dtype=complex)
        elif n_spin == 2:
            u_wfs = np.zeros(
                (*lambda_shape, n_kpts, n_states, n_orb, n_spin), dtype=complex
            )

        energies = np.zeros((*lambda_shape, n_kpts, n_states))

        for idx, param_set in enumerate(np.ndindex(*lambda_shape)):
            param_dict = {
                lambda_keys[i]: lambda_ranges[i][param_set] for i in range(dim_lambda)
            }

            # Generate the model with modified parameters
            modified_model = model_fxn(**param_dict)

            # Solve for eigenstates
            eigvals, eigvecs = modified_model.solve_ham(
                self.k_mesh.flat_mesh, return_eigvecs=True
            )

            # Store results
            energies[param_set] = eigvals
            u_wfs[param_set] = eigvecs

        # Reshape for compatibility with existing Berry curvature methods
        new_axes = (dim_lambda,) + tuple(range(dim_lambda)) + (dim_lambda + 1,)
        energies = np.transpose(energies, axes=new_axes)
        if self._nspin == 1:
            new_axes = (
                (dim_lambda,)
                + tuple(range(dim_lambda))
                + tuple(range(dim_lambda + 1, dim_lambda + 3))
            )
        else:
            new_axes = (
                (dim_lambda,)
                + tuple(range(dim_lambda))
                + tuple(range(dim_lambda + 1, dim_lambda + 4))
            )
        u_wfs = np.transpose(u_wfs, axes=new_axes)

        if self._nspin == 1:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_states)
        else:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_orb, n_spin)
        u_wfs = u_wfs.reshape(new_shape)
        energies = energies.reshape((*self.k_mesh.nks, *lambda_shape, n_states))

        self.set_wfs(u_wfs, cell_periodic=True)
        self.energies = energies
        self.is_energy_eigstate = True

    def solve_on_path(self, k_arr):
        """
        Solves on model passed when initialized. Not suitable for
        adiabatic parameters in the model beyond k.
        """
        eigvals, eigvecs = self.model.solve_ham(k_arr, return_eigvecs=True)
        self.set_wfs(eigvecs)
        self.energies = eigvals

    ###### Retrievers  #######

    def get_states(self, flatten_spin=False):
        """Returns dictionary containing Bloch and cell-periodic eigenstates."""
        assert hasattr(
            self, "_psi_wfs"
        ), "Need to call `solve_model` or `set_wfs` to initialize Bloch states"
        psi_wfs = self._psi_wfs
        u_wfs = self._u_wfs

        if flatten_spin:
            # shape is [nk1, ..., nkd, [n_lambda,] n_state, n_orb, n_spin], flatten last two axes
            psi_wfs = psi_wfs.reshape((*psi_wfs.shape[:-2], -1))
            u_wfs = u_wfs.reshape((*u_wfs.shape[:-2], -1))

        return {"Bloch": psi_wfs, "Cell periodic": u_wfs}

    def get_projector(self, return_Q=False):
        assert hasattr(
            self, "_P"
        ), "Need to call `solve_model` or `set_wfs` to initialize Bloch states"
        if return_Q:
            return self._P, self._Q
        else:
            return self._P

    def get_nbr_projector(self, return_Q=False):
        assert hasattr(
            self, "_P_nbr"
        ), "Need to call `solve_model` or `set_wfs` to initialize Bloch states"
        if return_Q:
            return self._P_nbr, self._Q_nbr
        else:
            return self._P_nbr

    def get_energies(self):
        assert hasattr(
            self, "energies"
        ), "Need to call `solve_model` to initialize energies"
        return self.energies

    def get_Bloch_Ham(self):
        """Returns the Bloch Hamiltonian of the model defined over the semi-full k-mesh."""
        if hasattr(self, "H_k"):
            return self.H_k
        else:
            self.set_Bloch_ham()
            return self.H_k

    def get_overlap_mat(self):
        """Returns overlap matrix.

        Overlap matrix defined as M_{n,m,k,b} = <u_{n, k} | u_{m, k+b}>
        """
        assert hasattr(
            self, "_M"
        ), "Need to call `solve_model` or `set_wfs` to initialize overlap matrix"
        return self._M

    def set_wfs(
        self, wfs, cell_periodic: bool = True, spin_flattened=False, set_projectors=True
    ):
        """
        Sets the Bloch and cell-periodic eigenstates as class attributes.

        Args:
            wfs (np.ndarray):
                Bloch (or cell-periodic) eigenstates defined on a semi-full k-mesh corresponding
                to nks passed during class instantiation. The mesh is assumed to exlude the
                endpoints, e.g. in reduced coordinates {k = [kx, ky, kz] | k_i in [0, 1)}.
        """
        if spin_flattened and self._nspin == 2:
            self._n_states = wfs.shape[-2]
        else:
            self._n_states = wfs.shape[self.state_axis]

        if self.dim_lam > 0:
            if self._nspin == 2:
                self._wf_shape = (
                    *self.nks,
                    *self.n_lambda,
                    self._n_states,
                    self._n_orb,
                    self._nspin,
                )
            else:
                self._wf_shape = (
                    *self.nks,
                    *self.n_lambda,
                    self._n_states,
                    self._n_orb,
                )
        else:
            if self._nspin == 2:
                self._wf_shape = (*self.nks, self._n_states, self._n_orb, self._nspin)
            else:
                self._wf_shape = (*self.nks, self._n_states, self._n_orb)

        wfs = wfs.reshape(self._wf_shape)

        if cell_periodic:
            self._u_wfs = wfs
            self._psi_wfs = self._apply_phase(wfs)
        else:
            self._psi_wfs = wfs
            self._u_wfs = self._apply_phase(wfs, inverse=True)

        if self.dim_lam == 0 and set_projectors:
            # overlap matrix
            self._M = self._get_self_overlap_mat()
            # band projectors
            self._set_projectors()

    def _get_pbc_wfs(self):

        dim_k = self.k_mesh.dim
        orb_vecs = self.model.get_orb_vecs(Cartesian=False)

        # Initialize the extended array by padding with an extra element along each k-axis
        pbc_uwfs = np.pad(
            self._u_wfs,
            pad_width=[
                (0, 1) if i < dim_k else (0, 0) for i in range(self._u_wfs.ndim)
            ],
            mode="wrap",
        )
        pbc_psiwfs = np.pad(
            self._psi_wfs,
            pad_width=[
                (0, 1) if i < dim_k else (0, 0) for i in range(self._psi_wfs.ndim)
            ],
            mode="wrap",
        )

        # Compute the reciprocal lattice vectors (unit vectors for each dimension)
        G_vectors = list(product([0, 1], repeat=dim_k))
        # Remove the zero vector
        G_vectors = [np.array(vector) for vector in G_vectors if any(vector)]

        for G in G_vectors:
            phase = np.exp(-1j * 2 * np.pi * (orb_vecs @ G.T)).T[np.newaxis, :]
            slices_new = []
            slices_old = []

            for i, value in enumerate(G):
                if value == 1:
                    slices_new.append(
                        slice(-1, None)
                    )  # Take the last element along this axis
                    slices_old.append(slice(0, None))
                else:
                    slices_new.append(slice(None))  # Take all elements along this axis
                    slices_old.append(slice(None))  # Take all elements along this axis

            # Add slices for any remaining dimensions (m, n) if necessary
            slices_new.extend([slice(None)] * (pbc_uwfs.ndim - len(G)))
            slices_old.extend([slice(None)] * (pbc_uwfs.ndim - len(G)))
            pbc_uwfs[tuple(slices_new)] *= phase

        return pbc_psiwfs, pbc_uwfs

    # Works with and without spin and lambda
    def _apply_phase(self, wfs, inverse=False):
        """
        Change between cell periodic and Bloch wfs by multiplying exp(\pm i k . tau)

        Args:
        wfs (pythtb.WFArray): Bloch or cell periodic wfs [k, nband, norb]

        Returns:
        wfsxphase (np.ndarray):
            wfs with orbitals multiplied by phase factor

        """
        lam = -1 if inverse else 1  # overall minus if getting cell periodic from Bloch
        per_dir = list(
            range(self.k_mesh.flat_mesh.shape[-1])
        )  # list of periodic dimensions
        # slice second dimension to only keep only periodic dimensions in orb
        per_orb = self.model.orb_vecs[:, per_dir]

        # compute a list of phase factors: exp(pm i k . tau) of shape [k_val, orbital]
        phases = np.exp(
            lam * 1j * 2 * np.pi * per_orb @ self.k_mesh.flat_mesh.T, dtype=complex
        ).T
        phases = phases.reshape(*self.k_mesh.nks, self._n_orb)

        if hasattr(self, "n_lambda") and self.n_lambda:
            phases = phases[..., np.newaxis, :]

        # if len(self._wf_shape) != len(wfs.shape):
        wfs = wfs.reshape(*self._wf_shape)

        # broadcasting to match dimensions
        if self._nspin == 1:
            # reshape to have each k-dimension as an axis
            # wfs = wfs.reshape(*self.k_mesh.nks, self._n_states, self._n_orb)
            # newaxis along state dimension
            phases = phases[..., np.newaxis, :]
        elif self._nspin == 2:
            # reshape to have each k-dimension as an axis
            # newaxis along state and spin dimension
            phases = phases[..., np.newaxis, :, np.newaxis]

        return wfs * phases

    # TODO: allow for projectors onto subbands
    # TODO: possibly get rid of nbr by storing boundary states
    def _set_projectors(self):
        num_nnbrs = self.k_mesh.num_nnbrs
        nnbr_idx_shell = self.k_mesh.nnbr_idx_shell

        if self._nspin == 2:
            u_wfs = self.get_states(flatten_spin=True)["Cell periodic"]
        else:
            u_wfs = self.get_states()["Cell periodic"]

        # band projectors
        self._P = np.einsum("...ni, ...nj -> ...ij", u_wfs, u_wfs.conj())
        self._Q = np.eye(self._n_orb * self._nspin) - self._P

        # NOTE: lambda friendly
        self._P_nbr = np.zeros(
            (self._P.shape[:-2] + (num_nnbrs,) + self._P.shape[-2:]), dtype=complex
        )
        self._Q_nbr = np.zeros_like(self._P_nbr)

        # NOTE: not lambda friendly
        # self._P_nbr = np.zeros((*nks, num_nnbrs, self._n_orb*self._nspin, self._n_orb*self._nspin), dtype=complex)
        # self._Q_nbr = np.zeros((*nks, num_nnbrs, self._n_orb*self._nspin, self._n_orb*self._nspin), dtype=complex)

        # TODO need shell to iterate over extra lambda dims also, shift accordingly
        for idx, idx_vec in enumerate(nnbr_idx_shell[0]):  # nearest neighbors
            # accounting for phase across the BZ boundary
            states_pbc = (
                np.roll(u_wfs, shift=tuple(-idx_vec), axis=self.k_axes)
                * self.k_mesh.bc_phase[..., idx, np.newaxis, :]
            )
            self._P_nbr[..., idx, :, :] = np.einsum(
                "...ni, ...nj -> ...ij", states_pbc, states_pbc.conj()
            )
            self._Q_nbr[..., idx, :, :] = (
                np.eye(self._n_orb * self._nspin) - self._P_nbr[..., idx, :, :]
            )

        return

    # TODO: allow for subbands and possible lamda dim
    def _get_self_overlap_mat(self):
        """Compute the overlap matrix of the cell periodic eigenstates.

        Overlap matrix of the form

        M_{m,n}^{k, k+b} = < u_{m, k} | u_{n, k+b} >

        Assumes that the last u_wf along each periodic direction corresponds to the
        next to last k-point in the mesh (excludes endpoints).

        Returns:
            M (np.array):
                Overlap matrix with shape [*nks, num_nnbrs, n_states, n_states]
        """

        # Assumes only one shell for now
        _, idx_shell = self.k_mesh.get_k_shell(N_sh=1, report=False)
        idx_shell = idx_shell[0]
        bc_phase = self.k_mesh.bc_phase

        # TODO: Not lambda friendly
        # overlap matrix
        M = np.zeros(
            (*self.k_mesh.nks, len(idx_shell), self._n_states, self._n_states),
            dtype=complex,
        )

        if self._nspin == 2:
            u_wfs = self.get_states(flatten_spin=True)["Cell periodic"]
        else:
            u_wfs = self.get_states()["Cell periodic"]

        for idx, idx_vec in enumerate(idx_shell):  # nearest neighbors
            # introduce phases to states when k+b is across the BZ boundary
            states_pbc = (
                np.roll(
                    u_wfs,
                    shift=tuple(-idx_vec),
                    axis=[i for i in range(self.k_mesh.dim)],
                )
                * bc_phase[..., idx, np.newaxis, :]
            )
            M[..., idx, :, :] = np.einsum(
                "...mj, ...nj -> ...mn", u_wfs.conj(), states_pbc
            )

        return M

    # TODO: Not working
    def berry_phase(self, dir=0, state_idx=None, evals=False):
        """
        Computes Berry phases for wavefunction arrays defined in parameter space.

        Parameters:
            wfs (np.ndarray):
                Wavefunction array of shape [*param_arr_lens, n_orb, n_orb] where
                axis -2 corresponds to the eigenvalue index and axis -1 corresponds
                to amplitude.
            dir (int):
                The direction (axis) in the parameter space along which to compute the Berry phase.

        Returns:
            phase (np.ndarray):
                Berry phases for the specified parameter space direction.
        """
        wfs = self.get_states()["Cell periodic"]
        if state_idx is not None:
            wfs = np.take(wfs, state_idx, axis=self.state_axis)
        orb_vecs = self.model.get_orb_vecs()
        dim_param = self.k_mesh.dim  # dimensionality of parameter space
        param_axes = np.arange(0, dim_param)  # parameter axes
        param_axes = np.setdiff1d(param_axes, dir)  # remove dir from axes to loop
        lens = [wfs.shape[ax] for ax in param_axes]  # sizes of loop directions
        idxs = np.ndindex(*lens)  # index mesh

        phase = np.zeros((*lens, wfs.shape[dim_param]))

        G = np.zeros(dim_param)
        G[0] = 1
        phase_shift = np.exp(-1j * 2 * np.pi * (orb_vecs @ G.T))
        print(param_axes)
        for idx_set in idxs:
            # print(idx_set)
            # take wfs along loop axis at given idex
            sliced_wf = wfs.copy()
            for ax, idx in enumerate(idx_set):
                # print(param_axes[ax])
                sliced_wf = np.take(sliced_wf, idx, axis=param_axes[ax])

            # print(sliced_wf.shape)
            end_state = sliced_wf[0, ...] * phase_shift[np.newaxis, :, np.newaxis]
            sliced_wf = np.append(sliced_wf, end_state[np.newaxis, ...], axis=0)
            phases = self.berry_loop(sliced_wf, evals=evals)
            phase[idx_set] = phases

        return phase

    # works in all cases
    def wilson_loop(self, wfs_loop, evals=False):
        """Compute Wilson loop unitary matrix and its eigenvalues for multiband Berry phases.

        Multiband Berry phases always returns numbers between -pi and pi.

        Args:
            wfs_loop (np.ndarray):
                Has format [loop_idx, band, orbital, spin] and loop has to be one dimensional.
                Assumes that first and last loop-point are the same. Therefore if
                there are n wavefunctions in total, will calculate phase along n-1
                links only!
            berry_evals (bool):
                If berry_evals is True then will compute phases for
                individual states, these corresponds to 1d hybrid Wannier
                function centers. Otherwise just return one number, Berry phase.
        """

        wfs_loop = wfs_loop.reshape(wfs_loop.shape[0], wfs_loop.shape[1], -1)
        ovr_mats = wfs_loop[:-1].conj() @ wfs_loop[1:].swapaxes(-2, -1)
        V, _, Wh = np.linalg.svd(ovr_mats, full_matrices=False)
        U_link = V @ Wh
        U_wilson = U_link[0]
        for i in range(1, len(U_link)):
            U_wilson = U_wilson @ U_link[i]

        # calculate phases of all eigenvalues
        if evals:
            evals = np.linalg.eigvals(U_wilson)  # Wilson loop eigenvalues
            eval_pha = -np.angle(evals)  # Multiband  Berrry phases
            return U_wilson, eval_pha
        else:
            return U_wilson


    # works in all cases
    def berry_loop(self, wfs_loop, evals=False):
        U_wilson = self.wilson_loop(wfs_loop, evals=evals)

        if evals:
            return U_wilson[1]
        else:
            return -np.angle(np.linalg.det(U_wilson))  # total Berry phase


    # Works in all cases
    def get_links(self, state_idx):
        wfs = self.get_states()["Cell periodic"]

        orb_vecs = self.model.orb_vecs  # Orbital position vectors (reduced units)
        n_param = self.n_adia  # Number of points in adiabatic mesh
        dim = self.dim_adia  # Total dimensionality of adiabatic space
        n_spin = getattr(self, "_nspin", 1)  # Number of spin components

        # State selection
        if state_idx is not None:
            wfs = np.take(wfs, state_idx, axis=self.state_axis)
            if isinstance(state_idx, int):
                wfs = np.expand_dims(wfs, self.state_axis)

        n_states = wfs.shape[self.state_axis]

        U_forward = []
        wfs_flat = wfs.reshape(*n_param, n_states, -1)
        for mu in range(dim):
            # print(f"Computing links for direction: mu={mu}")
            wfs_shifted = np.roll(wfs, -1, axis=mu)

            # Apply phase factor e^{-i G.r} to shifted u_nk states at the boundary (now 0th state)
            if mu < self.k_mesh.dim:
                mask = np.zeros(n_param, dtype=bool)
                idx = [slice(None)] * dim
                idx[mu] = n_param[mu] - 1
                mask[tuple(idx)] = True

                G = np.zeros(self.k_mesh.dim)
                G[mu] = 1
                phase = np.exp(-2j * np.pi * G @ orb_vecs.T)

                if n_spin == 1:
                    phase_broadcast = phase[np.newaxis, :]
                    mask_expanded = mask[..., np.newaxis, np.newaxis]
                else:
                    phase_broadcast = phase[np.newaxis, :, np.newaxis]
                    mask_expanded = mask[..., np.newaxis, np.newaxis, np.newaxis]

                wfs_shifted = np.where(
                    mask_expanded, wfs_shifted * phase_broadcast, wfs_shifted
                )

            # Flatten along spin
            wfs_shifted_flat = wfs_shifted.reshape(*n_param, n_states, -1)
            # <u_nk| u_m k+delta_mu>
            ovr_mu = wfs_flat.conj() @ wfs_shifted_flat.swapaxes(-2, -1)

            U_forward_mu = np.zeros_like(ovr_mu, dtype=complex)
            V, _, Wd = np.linalg.svd(ovr_mu, full_matrices=False)
            U_forward_mu = V @ Wd
            U_forward.append(U_forward_mu)

        return np.array(U_forward)


    def berry_flux_plaq(self, state_idx=None, non_abelian=False):
        """Compute fluxes on a two-dimensional plane of states.

        For a given set of states, returns the band summed Berry curvature
        rank-2 tensor for all combinations of surfaces in reciprocal space.
        By convention, the Berry curvature is reported at the point where the loop
        started, which is the lower left corner of a plaquette.
        """
        n_states = len(state_idx)  # Number of states considered
        n_param = self.n_adia  # Number of points in adiabatic mesh
        dim = self.dim_adia  # Total dimensionality of adiabatic space

        # Initialize Berry flux array
        shape = (
            (dim, dim, *n_param, n_states, n_states)
            if non_abelian
            else (dim, dim, *n_param)
        )
        Berry_flux = np.zeros(shape, dtype=complex)

        # Overlaps <u_{nk} | u_{n, k+delta k_mu}>
        U_forward = self.get_links(state_idx=state_idx)
        # Wilson loops W = U_{mu}(k_0) U_{nu}(k_0 + delta_mu) U^{-1}_{mu}(k_0 + delta_mu + delta_nu) U^{-1}_{nu}(k_0)
        for mu in range(dim):
            for nu in range(mu + 1, dim):
                print(f"Computing flux in plane: mu={mu}, nu={nu}")
                U_mu = U_forward[mu]
                U_nu = U_forward[nu]

                U_nu_shift_mu = np.roll(U_nu, -1, axis=mu)
                U_mu_shift_nu = np.roll(U_mu, -1, axis=nu)

                U_wilson = np.matmul(
                    np.matmul(
                        np.matmul(U_mu, U_nu_shift_mu),
                        U_mu_shift_nu.conj().swapaxes(-1, -2),
                    ),
                    U_nu.conj().swapaxes(-1, -2),
                )

                if non_abelian:
                    eigvals, eigvecs = np.linalg.eig(U_wilson)
                    angles = -np.angle(eigvals)
                    angles_diag = np.einsum(
                        "...i, ij -> ...ij", angles, np.eye(angles.shape[-1])
                    )
                    eigvecs_inv = np.linalg.inv(eigvecs)
                    phases_plane = np.matmul(
                        np.matmul(eigvecs, angles_diag), eigvecs_inv
                    )
                else:
                    det_U = np.linalg.det(U_wilson)
                    phases_plane = -np.angle(det_U)

                Berry_flux[mu, nu] = phases_plane
                Berry_flux[nu, mu] = -phases_plane

        return Berry_flux


    def berry_curv(
        self,
        dirs=None,
        state_idx=None,
        non_abelian=False,
        delta_lam=1,
        return_flux=False,
        Kubo=False,
    ):

        nks = self.nks  # Number of mesh points per direction
        n_lambda = self.n_lambda
        dim_k = len(nks)  # Number of k-space dimensions
        dim_lam = len(n_lambda)  # Number of adiabatic dimensions
        dim_total = dim_k + dim_lam  # Total number of dimensions

        if dim_k < 2:
            raise ValueError("Berry curvature only defined for dim_k >= 2.")

        if Kubo:
            if not self.is_energy_eigstate:
                raise ValueError("Must be energy eigenstate to use Kubo formula.")
            if not hasattr(self, "_u_wfs") or not hasattr(self, "energies"):
                raise ValueError(
                    "Must diagonalize model first to set wavefunctions and energies."
                )
            if state_idx is not None:
                print(
                    "Berry curvature in Kubo formula is for all occupied bands. Using half filling for occupied bands."
                )
            if dim_lam != 0 or delta_lam != 1:
                raise ValueError(
                    "Adiabatic dimensions not yet supported for Kubo formula."
                )
            if return_flux:
                print(
                    "Kubo formula doesn't support flux. Will return dimensionful Berry curvature only."
                )

            u_wfs = self.get_states(flatten_spin=True)["Cell periodic"]
            energies = self.energies
            # flatten k_dims
            u_wfs = u_wfs.reshape(-1, u_wfs.shape[-2], u_wfs.shape[-1])
            energies = energies.reshape(-1, energies.shape[-1])
            n_states = u_wfs.shape[-2]

            if n_states != self.model.nstate:
                raise ValueError(
                    "Wavefunctions must be defined for all bands, not just a subset."
                )

            k_mesh = self.k_mesh.flat_mesh
            occ_idx = np.arange(n_states // 2)
            abelian = not non_abelian
            if dirs is None:
                dirs = "all"
                b_curv = self.model.berry_curvature(
                    k_mesh,
                    evals=energies,
                    evecs=u_wfs,
                    occ_idxs=occ_idx,
                    abelian=abelian,
                )
                b_curv = b_curv.reshape(*b_curv.shape[:2], *nks, *b_curv.shape[3:])
            else:
                b_curv = self.model.berry_curvature(
                    k_mesh,
                    evals=energies,
                    evecs=u_wfs,
                    occ_idxs=occ_idx,
                    abelian=abelian,
                    dirs=dirs,
                )
                b_curv = b_curv.reshape(*nks, *b_curv.shape[3:])

            return b_curv

        Berry_flux = self.berry_flux_plaq(state_idx=state_idx, non_abelian=non_abelian)
        Berry_curv = np.zeros_like(Berry_flux, dtype=complex)

        dim = Berry_flux.shape[0]  # Number of dimensions in parameter space
        recip_lat_vecs = (
            self.model.get_recip_lat()
        )  # Expressed in cartesian (x,y,z) coordinates

        dks = np.zeros((dim_total, dim_total))
        dks[:dim_k, :dim_k] = recip_lat_vecs / np.array(self.nks)[:, None]
        if self.dim_lam > 0:
            np.fill_diagonal(dks[dim_k:, dim_k:], delta_lam / np.array(self.n_lambda))

        # Divide by area elements for the (mu, nu)-plane
        for mu in range(dim):
            for nu in range(mu + 1, dim):
                A = np.vstack([dks[mu], dks[nu]])
                # area_element = np.prod([np.linalg.norm(dk[i]), np.linalg.norm(dk[j])])
                area_element = np.sqrt(np.linalg.det(A @ A.T))

                # Divide flux by the area element to get approx curvature
                Berry_curv[mu, nu] = Berry_flux[mu, nu] / area_element
                Berry_curv[nu, mu] = Berry_flux[nu, mu] / area_element

        if dirs is not None:
            Berry_curv, Berry_flux = Berry_curv[dirs], Berry_flux[dirs]
        if return_flux:
            return Berry_curv, Berry_flux
        else:
            return Berry_curv


    def chern_num(self, dirs=(0, 1), band_idxs=None):
        if band_idxs is None:
            n_occ = int(self._n_states / 2)
            band_idxs = np.arange(n_occ)  # assume half-filled occupied

        berry_flux = self.berry_flux_plaq(state_idx=band_idxs)
        Chern = np.sum(berry_flux[dirs] / (2 * np.pi))

        return Chern


    # TODO allow for subbands
    def trace_metric(self):
        P = self._P
        Q_nbr = self._Q_nbr

        nks = Q_nbr.shape[:-3]
        num_nnbrs = Q_nbr.shape[-3]
        w_b, _, _ = self.k_mesh.get_weights(N_sh=1)

        T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)
        for nbr_idx in range(num_nnbrs):  # nearest neighbors
            T_kb[..., nbr_idx] = np.trace(
                P[..., :, :] @ Q_nbr[..., nbr_idx, :, :], axis1=-1, axis2=-2
            )

        return w_b[0] * np.sum(T_kb, axis=-1)


    # TODO allow for subbands
    def omega_til(self):
        M = self._M
        w_b, k_shell, idx_shell = self.k_mesh.get_weights(N_sh=1)
        w_b = w_b[0]
        k_shell = k_shell[0]

        nks = M.shape[:-3]
        Nk = np.prod(nks)
        k_axes = tuple([i for i in range(len(nks))])

        diag_M = np.diagonal(M, axis1=-1, axis2=-2)
        log_diag_M_imag = np.log(diag_M).imag
        abs_diag_M_sq = abs(diag_M) ** 2

        r_n = -(1 / Nk) * w_b * np.sum(log_diag_M_imag, axis=k_axes).T @ k_shell

        Omega_tilde = (
            (1 / Nk)
            * w_b
            * (
                np.sum((-log_diag_M_imag - k_shell @ r_n.T) ** 2)
                + np.sum(abs(M) ** 2)
                - np.sum(abs_diag_M_sq)
            )
        )
        return Omega_tilde


    def interp_op(self, O_k, k_path, plaq=False):
        k_mesh = np.copy(self.k_mesh.square_mesh)
        k_idx_arr = self.k_mesh.idx_arr
        nks = self.k_mesh.nks
        dim_k = len(nks)
        Nk = np.prod([nks])

        supercell = list(
            product(
                *[range(-int((nk - nk % 2) / 2), int((nk - nk % 2) / 2)) for nk in nks]
            )
        )

        if plaq:
            # shift by half a mesh point to get the center of the plaquette
            k_mesh += np.array([(1 / nk) / 2 for nk in nks])

        # Fourier transform to real space
        O_R = np.zeros((len(supercell), *O_k.shape[dim_k:]), dtype=complex)
        for idx, pos in enumerate(supercell):
            for k_idx in k_idx_arr:
                R_vec = np.array(pos)
                phase = np.exp(-1j * 2 * np.pi * np.vdot(k_mesh[k_idx], R_vec))
                O_R[idx] += O_k[k_idx] * phase / Nk

        # interpolate to arbitrary k
        O_k_interp = np.zeros((k_path.shape[0], *O_k.shape[dim_k:]), dtype=complex)
        for k_idx, k in enumerate(k_path):
            for idx, pos in enumerate(supercell):
                R_vec = np.array(pos)
                phase = np.exp(1j * 2 * np.pi * np.vdot(k, R_vec))
                O_k_interp[k_idx] += O_R[idx] * phase

        return O_k_interp


    def interp_energy(self, k_path, return_eigvecs=False):
        H_k_proj = self.get_proj_ham()
        H_k_interp = self.interp_op(H_k_proj, k_path)
        if return_eigvecs:
            u_k_interp = self.interp_op(self._u_wfs, k_path)
            eigvals_interp, eigvecs_interp = np.linalg.eigh(H_k_interp)
            eigvecs_interp = np.einsum(
                "...ij, ...ik -> ...jk", u_k_interp, eigvecs_interp
            )
            eigvecs_interp = np.transpose(eigvecs_interp, axes=[0, 2, 1])
            return eigvals_interp, eigvecs_interp
        else:
            eigvals_interp = np.linalg.eigvalsh(H_k_interp)
            return eigvals_interp


    # TODO allow for subbands
    def get_proj_ham(self):
        if not hasattr(self, "H_k_proj"):
            self.set_Bloch_ham()
        H_k_proj = self.u_wfs.conj() @ self.H_k @ np.swapaxes(self.u_wfs, -1, -2)
        return H_k_proj
