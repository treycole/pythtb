import numpy as np
from itertools import combinations_with_replacement as comb
from itertools import product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tb_model import TBModel

__all__ = [
    "k_uniform_mesh",
    "k_path",
    "Mesh",
]


def k_uniform_mesh(model, mesh_size) -> np.ndarray:
    r"""
    Returns a uniform grid of k-points that can be passed to
    passed to function :func:`pythtb.tb_model.solve_all`. This
    function is useful for plotting density of states histogram
    and similar.

    Returned uniform grid of k-points always contains the origin.

    :param mesh_size: Number of k-points in the mesh in each
        periodic direction of the model.

    :returns:

        * **k_vec** -- Array of k-vectors on the mesh that can be
        directly passed to function  :func:`pythtb.tb_model.solve_all`.

    Example usage::

        # returns a 10x20x30 mesh of a tight binding model
        # with three periodic directions
        k_vec = my_model.k_uniform_mesh([10,20,30])
        # solve model on the uniform mesh
        my_model.solve_all(k_vec)

    """

    # get the mesh size and checks for consistency
    use_mesh = np.array(list(map(round, mesh_size)), dtype=int)
    if use_mesh.shape != (model.dim_k,):
        print(use_mesh.shape)
        raise Exception("\n\nIncorrect size of the specified k-mesh!")
    if np.min(use_mesh) <= 0:
        raise Exception("\n\nMesh must have positive non-zero number of elements.")

    axes = [np.linspace(0, 1, n, endpoint=False) for n in use_mesh]
    mesh = np.meshgrid(*axes, indexing="ij")
    k_vec = np.stack(mesh, axis=-1).reshape(-1, len(use_mesh))

    return k_vec


def k_path(model, kpts, nk, report=True):
    r"""

    Interpolates a path in reciprocal space between specified
    k-points.  In 2D or 3D the k-path can consist of several
    straight segments connecting high-symmetry points ("nodes"),
    and the results can be used to plot the bands along this path.

    The interpolated path that is returned contains as
    equidistant k-points as possible.

    :param kpts: Array of k-vectors in reciprocal space between
        which interpolated path should be constructed. These
        k-vectors must be given in reduced coordinates.  As a
        special case, in 1D k-space kpts may be a string:

        * *"full"*  -- Implies  *[ 0.0, 0.5, 1.0]*  (full BZ)
        * *"fullc"* -- Implies  *[-0.5, 0.0, 0.5]*  (full BZ, centered)
        * *"half"*  -- Implies  *[ 0.0, 0.5]*  (half BZ)

    :param nk: Total number of k-points to be used in making the plot.

    :param report: Optional parameter specifying whether printout
        is desired (default is True).

    :returns:

        * **k_vec** -- Array of (nearly) equidistant interpolated
        k-points. The distance between the points is calculated in
        the Cartesian frame, however coordinates themselves are
        given in dimensionless reduced coordinates!  This is done
        so that this array can be directly passed to function
        :func:`pythtb.tb_model.solve_all`.

        * **k_dist** -- Array giving accumulated k-distance to each
        k-point in the path.  Unlike array *k_vec* this one has
        dimensions! (Units are defined here so that for an
        one-dimensional crystal with lattice constant equal to for
        example *10* the length of the Brillouin zone would equal
        *1/10=0.1*.  In other words factors of :math:`2\pi` are
        absorbed into *k*.) This array can be used to plot path in
        the k-space so that the distances between the k-points in
        the plot are exact.

        * **k_node** -- Array giving accumulated k-distance to each
        node on the path in Cartesian coordinates.  This array is
        typically used to plot nodes (typically special points) on
        the path in k-space.

    Example usage::

        # Construct a path connecting four nodal points in k-space
        # Path will contain 401 k-points, roughly equally spaced
        path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]]
        (k_vec,k_dist,k_node) = my_model.k_path(path,401)
        # solve for eigenvalues on that path
        evals = tb.solve_all(k_vec)
        # then use evals, k_dist, and k_node to plot bandstructure
        # (see examples)

    """
    dim = model.dim_k

    # Parse kpts and validate
    k_list = _parse_kpts(kpts, dim)
    if k_list.shape[1] != dim:
        raise ValueError(
            f"Dimension mismatch: kpts shape {k_list.shape}, model dim {dim}"
        )
    if nk < len(k_list):
        raise ValueError("nk must be >= number of nodes in kpts")

    # Extract periodic lattice and compute k-space metric
    lat_per = model.lat_vecs[model.per]
    k_metric = np.linalg.inv(lat_per @ lat_per.T)

    # Compute segment vectors and lengths in Cartesian metric
    diffs = k_list[1:] - k_list[:-1]
    seg_lengths = np.sqrt(np.einsum("ij,ij->i", diffs @ k_metric, diffs))

    # Accumulated node distances
    k_node = np.concatenate(([0.0], np.cumsum(seg_lengths)))

    # Determine indices in the final array corresponding to each node
    node_index = np.rint(k_node / k_node[-1] * (nk - 1)).astype(int)

    # Initialize output arrays
    k_vec = np.empty((nk, dim))
    k_dist = np.empty(nk)

    # Interpolate each segment
    for i, (start, end) in enumerate(zip(node_index[:-1], node_index[1:])):
        length = end - start
        t = np.linspace(0, 1, length + 1)
        k_vec[start : end + 1] = k_list[i] + np.outer(t, diffs[i])
        k_dist[start : end + 1] = k_node[i] + t * seg_lengths[i]

    # Trim any round-off overshoot
    k_vec = k_vec[:nk]
    k_dist = k_dist[:nk]

    if report:
        _report(k_list, lat_per, k_metric, k_node, node_index)

    return k_vec, k_dist, k_node


def _parse_kpts(kpts, dim):
    """
    Parse special string cases for 1D and ensure array shape (n_nodes, dim).
    """
    if isinstance(kpts, str) and dim == 1:
        presets = {
            "full": [[0.0], [0.5], [1.0]],
            "fullc": [[-0.5], [0.0], [0.5]],
            "half": [[0.0], [0.5]],
        }
        return np.array(presets[kpts], float)

    arr = np.array(kpts, float)
    if arr.ndim == 1 and dim == 1:
        arr = arr[:, None]
    return arr


def _interpolate_path(nodes: np.ndarray, n_interp: int) -> np.ndarray:
    """
    Given `nodes` shape (R, D), returns a linear interpolation
    along each consecutive pair, totalling R*n_interp points.
    """
    segments = []
    for i in range(len(nodes)-1):
        start, end = nodes[i], nodes[i+1]
        t = np.linspace(0,1,n_interp,endpoint=False)
        segments.append(start[None,:] + (end-start)[None,:]*t[:,None])
    # add the final node
    segments.append(nodes[-1:,:])
    return np.vstack(segments)


def _report(k_list, lat_per, k_metric, k_node, node_index):
    """
    Print a concise report of the k-path construction, including
    segment distances and the start/end node coordinates.
    """
    print("----- k_path report -----")
    np.set_printoptions(precision=5)
    print("Real-space lattice vectors:\n", lat_per)
    print("K-space metric tensor:\n", k_metric)
    print("Nodes (reduced coords):\n", k_list)
    if lat_per.shape[0] == lat_per.shape[1]:
        gvecs = np.linalg.inv(lat_per).T
        print("Reciprocal-space vectors:\n", gvecs)
        print("Nodes (Cartesian coords):\n", k_list @ gvecs)

    print("Segments:")
    for n in range(1, len(k_node)):
        length = k_node[n] - k_node[n - 1]
        print(
            f"  Node {n-1} {k_list[n-1]} to Node {n} {k_list[n]}: "
            f"distance = {length:.5f}"
        )

    print("Node distances (cumulative):", k_node)
    print("Node indices in path:", node_index)
    print("-------------------------")

class Mesh:
    def __init__(
        self,
        model: "TBModel",
        dim_k: int,
        dim_param: int,
        shape_k  = None,
        shape_param = None,
        path_k:  np.ndarray = None,   # (N_k_nodes, dim_k)
        path_param: np.ndarray = None, # (N_p_nodes, dim_param)
        n_interp: int = None
    ):
        r"""
        You must supply *either* a full-grid (shape_k & shape_param), 
        *or* a piecewise path (path_k and/or path_param) + interpolation count.

        - Full grid:
            path_* is None, shape_* given.
        - Pure k-path:
            path_k given, n_interp, shape_param=None.
        - Pure param-path:
            path_param given, n_interp, shape_k=None.
        - Mixed path through :math:`(\mathbf{k}, \lambda)`-space :
            both path_k & path_param given, n_interp.

        After init, `.grid` will be
          • shape (*shape_k, *shape_param, dim_k+dim_param) for full-grid,  
          • shape (N_nodes·n_interp, dim_k+dim_param) for any path.  
        """
        self.model      = model
        self.dim_k      = dim_k
        self.dim_param  = dim_param

        # Decide full-grid vs path:
        is_full  = (shape_k is not None or shape_param is not None) and (path_k is path_param is None)
        is_path  = (path_k is not None or path_param is not None)

        if is_full:
            # build 0→1 linspaces
            k_axes = [ np.linspace(0,1,n,endpoint=True) for n in (shape_k or []) ]
            p_axes = [ np.linspace(0,1,m,endpoint=True) for m in (shape_param or []) ]
            # meshgrid
            mesh = np.meshgrid(*k_axes, *p_axes, indexing="ij")
            # each mesh[i] gives coordinates along axis i; stack last axis
            self.grid = np.stack(mesh, axis=-1)   # shape = (*shape_k, *shape_param, dim_k+dim_param)
            self.shape_k     = tuple(shape_k or [])
            self.shape_param = tuple(shape_param or [])

        elif is_path:
            # build piecewise nodes for k and/or param
            # e.g. for k-path: path_k: (R, dim_k), n_interp → R·n_interp points
            # we interpolate each segment linearly
            coords_k = _interpolate_path(path_k,   n_interp) if path_k   is not None else np.zeros((n_interp,0))
            coords_p = _interpolate_path(path_param,n_interp) if path_param is not None else np.zeros((n_interp,0))
            # tile to match mixed dims: if both present, tile coords_k for every λ and vice versa
            if path_k is not None and path_param is not None:
                # coords_k: (Rk, dim_k), coords_p: (Rp, dim_param)
                k_rep = np.repeat(coords_k[:,None,:],  coords_p.shape[0], axis=1)  # (Rk,Rp,dim_k)
                p_rep = np.repeat(coords_p[None,:,:],  coords_k.shape[0], axis=0)  # (Rk,Rp,dim_param)
                stacked = np.concatenate([k_rep, p_rep], axis=-1)                  # (Rk,Rp,dim_k+dim_param)
                self.grid = stacked.reshape(-1, dim_k+dim_param)

            else:
                # pure k or pure λ
                self.grid = np.concatenate([coords_k, coords_p], axis=1)  # (N, dim_k+dim_param)

            self.shape_k     = ()
            self.shape_param = ()

        else:
            raise ValueError("Must specify either full-grid shapes or path nodes + n_interp")

        # Flatten vs structured
        self.flat    = self.grid.reshape(-1, dim_k + dim_param)
        self.axis_types = ['k']*dim_k + ['param']*dim_param
        self.k_axes     = [i for i,t in enumerate(self.axis_types) if t=='k']
        self.p_axes     = [i for i,t in enumerate(self.axis_types) if t=='param']

        # Precompute any k-space phases only if dim_k > 0
        if self.dim_k:
            self.recip_lat_vecs = model.get_recip_lat()
            self.orb_phases     = self.get_orb_phases()
            self.bc_phase       = self.get_boundary_phase()

    def gen_k_mesh(
        self, centered: bool = False, flat: bool = True, endpoint: bool = False
    ) -> np.ndarray:
        """Generate a regular k-mesh in reduced coordinates.

        Args:
            centered (bool):
                If True, mesh is defined from [-0.5, 0.5] along each direction.
                Defaults to False.
            flat (bool):
                If True, returns flattened array of k-points (e.g. of dimension nkx*nky*nkz x 3).
                If False, returns reshaped array with axes along each k-space dimension
                (e.g. of dimension nkx x nky x nkz x 3). Defaults to True.
            endpoint (bool):
                If True, includes 1 (edge of BZ in reduced coordinates) in the mesh. Defaults to False. When Wannierizing shoule

        Returns:
            k-mesh (np.ndarray):
                Array of k-mesh coordinates.
        """

        end_pts = [-0.5, 0.5] if centered else [0, 1]
        k_vals = [
            np.linspace(end_pts[0], end_pts[1], nk, endpoint=endpoint)
            for nk in self.nks
        ]
        flat_mesh = np.stack(np.meshgrid(*k_vals, indexing="ij"), axis=-1)

        return flat_mesh if not flat else flat_mesh.reshape(-1, len(k_vals))

    def get_k_shell(self, N_sh: int, report: bool = False):
        """Generates shells of k-points around the Gamma point.

        Returns array of vectors connecting the origin to nearest neighboring k-points
        in the mesh, along with vectors of reduced coordinates.

        Args:
            N_sh (int):
                Number of nearest neighbor shells.
            report (bool):
                If True, prints a summary of the k-shell.

        Returns:
            k_shell (np.ndarray[float]):
                Array of vectors in inverse units of lattice vectorsconnecting nearest neighbor k-mesh points.
            idx_shell (np.ndarray[int]):
                Array of vectors of integers used for indexing the nearest neighboring k-mesh points
                to a given k-mesh point.
        """
        recip_lat_vecs = self.recip_lat_vecs
        # basis vectors connecting neighboring mesh points (in inverse Cartesian units)
        dk = np.array([recip_lat_vecs[i] / nk for i, nk in enumerate(self.nks)])
        # array of integers e.g. in 2D for N_sh = 1 would be [0,1], [1,0], [0,-1], [-1,0]
        nnbr_idx = list(product(list(range(-N_sh, N_sh + 1)), repeat=self.dim))
        nnbr_idx.remove((0,) * self.dim)
        nnbr_idx = np.array(nnbr_idx)
        # vectors connecting k-points near Gamma point (in inverse lattice vector units)
        b_vecs = np.array([nnbr_idx[i] @ dk for i in range(nnbr_idx.shape[0])])
        # distances to points around Gamma
        dists = np.array(
            [np.vdot(b_vecs[i], b_vecs[i]) for i in range(b_vecs.shape[0])]
        )
        # remove numerical noise
        dists = dists.round(10)

        # sorting by distance
        sorted_idxs = np.argsort(dists)
        dists_sorted = dists[sorted_idxs]
        b_vecs_sorted = b_vecs[sorted_idxs]
        nnbr_idx_sorted = nnbr_idx[sorted_idxs]

        unique_dists = sorted(list(set(dists)))  # removes repeated distances
        keep_dists = unique_dists[:N_sh]  # keep only distances up to N_sh away
        # keep only b_vecs in N_sh shells
        k_shell = [
            b_vecs_sorted[np.isin(dists_sorted, keep_dists[i])]
            for i in range(len(keep_dists))
        ]
        idx_shell = [
            nnbr_idx_sorted[np.isin(dists_sorted, keep_dists[i])]
            for i in range(len(keep_dists))
        ]

        if report:
            dist_degen = {ud: len(k_shell[i]) for i, ud in enumerate(keep_dists)}
            print("k-shell report:")
            print("--------------")
            print(f"Reciprocal lattice vectors: {self._recip_vecs}")
            print(f"Distances and degeneracies: {dist_degen}")
            print(f"k-shells: {k_shell}")
            print(f"idx-shells: {idx_shell}")

        return k_shell, idx_shell

    def get_weights(self, N_sh=1, report=False):
        """Generates the finite difference weights on a k-shell."""
        k_shell, idx_shell = self.get_k_shell(N_sh=N_sh, report=report)
        dim_k = len(self.nks)
        Cart_idx = list(comb(range(dim_k), 2))
        n_comb = len(Cart_idx)

        A = np.zeros((n_comb, N_sh))
        q = np.zeros((n_comb))

        for j, (alpha, beta) in enumerate(Cart_idx):
            if alpha == beta:
                q[j] = 1
            for s in range(N_sh):
                b_star = k_shell[s]
                for i in range(b_star.shape[0]):
                    b = b_star[i]
                    A[j, s] += b[alpha] * b[beta]

        U, D, Vt = np.linalg.svd(A, full_matrices=False)
        w = (Vt.T @ np.linalg.inv(np.diag(D)) @ U.T) @ q
        if report:
            print(f"Finite difference weights: {w}")
        return w, k_shell, idx_shell

    def get_boundary_phase(self):
        """
        Get phase factors to multiply the cell periodic states in the first BZ
        related by the pbc u_{n, k+G} = u_{n, k} exp(-i G . r)

        Returns:
            bc_phase (np.ndarray):
                The shape is [...k(s), shell_idx] where shell_idx is an integer
                corresponding to a particular idx_vec where the convention is to go
                counter-clockwise (e.g. square lattice 0 --> [1, 0], 1 --> [0, 1] etc.)

        """
        # --- unpack everything ---
        nks = np.array(self.nks)  # (dim,)
        orb_vecs = self.model.orb_vecs  # (n_orb, dim)
        nbrs = np.array(self.nnbr_idx_shell[0])  # (N_nbr, dim)
        idx_arr = self.idx_arr  # (Nk, dim)
        Nk, dim = idx_arr.shape
        N_nbr = nbrs.shape[0]
        n_orb = orb_vecs.shape[0]
        nspin = self.model.nspin

        # --- compute neighbor indices and how many cells we jumped over ---
        shifted_idx = idx_arr[:, None, :] + nbrs[None, :, :]  # (Nk, N_nbr, dim)
        mask_pos = shifted_idx >= nks  # True where you stepped >= +1 cell
        mask_neg = shifted_idx < 0  # True where you stepped <= –1 cell
        cross = (mask_pos | mask_neg).any(axis=2)  # (Nk, N_nbr)
        G = mask_pos.astype(np.int8) - mask_neg.astype(np.int8)

        # build output filled with 1’s
        if nspin == 1:
            bc_shape = (Nk, N_nbr, n_orb)
        elif nspin == 2:
            bc_shape = (Nk, N_nbr, n_orb, 2)
        else:
            raise ValueError("Wrong spin value, must be either 1 or 2")
        bc = np.ones(bc_shape, complex)

        # only for the True positions compute phase
        ki, ni = np.nonzero(cross)  # lists of length M << Nk*N_nbr
        if ki.size:
            # extract only the G’s at those positions: (M, dim)
            Gc = G[ki, ni, :]  # shape (M, dim)

            # dot with each orbital coordinate: (M, n_orb)
            dot = Gc.dot(orb_vecs.T)

            # complex phase for each of those M×n_orb points
            phase = np.exp(-2j * np.pi * dot)  # shape (M, n_orb)
            if nspin == 1:
                bc[ki, ni] = phase
            elif nspin == 2:
                bc[ki, ni] = phase[..., None]

        return bc.reshape(*nks, N_nbr, n_orb * nspin)

    def get_orb_phases(self, inverse=False):
        """Returns exp(\pm i k.tau) factors

        Args:
            Inverse (bool):
                If True, multiplies factor of -1 for mutiplying Bloch states to get cell-periodic states.
        """
        lam = -1 if inverse else 1  # overall minus if getting cell periodic from Bloch
        per_dir = list(range(self.flat_mesh.shape[-1]))  # list of periodic dimensions
        # slice second dimension to only keep only periodic dimensions in orb
        per_orb = self.model.orb_vecs[:, per_dir]

        # compute a list of phase factors [k_val, orbital]
        wf_phases = np.exp(
            lam * 1j * 2 * np.pi * per_orb @ self.flat_mesh.T, dtype=complex
        ).T
        return wf_phases  # 1D numpy array of dimension norb