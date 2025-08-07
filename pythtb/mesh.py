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


"""
 def __init__(self, model: "TBModel", *nks):
        self.model = model
        self.nks = nks
        self.Nk = np.prod(nks)
        self.dim: int = len(nks)
        self.recip_lat_vecs = model.get_recip_lat()
        idx_grid = np.indices(nks, dtype=int)
        idx_arr = idx_grid.reshape(len(nks), -1).T
        self.idx_arr: list = idx_arr  # 1D list of all k_indices (integers)
        self.square_mesh: np.ndarray = self.gen_k_mesh(
            flat=False, endpoint=False
        )  # each index is a direction in k-space
        self.flat_mesh: np.ndarray = self.gen_k_mesh(
            flat=True, endpoint=False
        )  # 1D list of k-vectors

        # nearest neighbor k-shell
        self.nnbr_w_b, _, self.nnbr_idx_shell = self.get_weights(N_sh=1)
        self.num_nnbrs = len(self.nnbr_idx_shell[0])

        # matrix of e^{-i G . r} phases
        self.bc_phase = self.get_boundary_phase()
        self.orb_phases = self.get_orb_phases()
"""

class Mesh:
    def __init__(self,  model: "TBModel"):
        """Initialize a Mesh object for a given TBModel.

        This class is responsible for constructing the mesh in k-space and parameter space.
        It provides methods to build both grid and path representations of the mesh.

        After calling :meth:`build_path` or :meth:`build_grid`, the mesh will have the following shapes:

        - ``.grid`` has shape ``(*shape_k, *shape_param, dim_k+dim_param)``

        - ``.flat`` always has shape ``(N_points, dim_k+dim_param)``

        - ``.k_axes`` and ``.param_axes`` tell you which indices in flat or the last axis of grid correspond to k vs. parameter dims.

        Parameters
        ----------
        model : TBModel
            The tight-binding model to use.
        """

        self._model = model
        self._dim_k = model.dim_k
        self._dim_param = None
        self._grid = None
        self._flat = None
        self._k_axes = None
        self._param_axes = None

    @property
    def model(self):
        return self._model 

    @property
    def points(self):
        return self._points

    @property
    def flat(self):
        return self._points
    
    @property
    def grid(self):
        shape_k = self.shape_k if self.shape_k is not None else []
        shape_param = self.shape_param if self.shape_param is not None else []
        return self._points.reshape(*shape_k, *shape_param, self.dim_k + self.dim_param)
    
    @property
    def k_axes(self):
        return self._k_axes

    @property
    def param_axes(self):
        return self._param_axes
    
    @property
    def shape_k(self):
        return self._shape_k

    @property
    def shape_param(self):
        return self._shape_param

    @property
    def dim_param(self):
        return self._dim_param

    @property
    def dim_k(self):
        return self._dim_k
    

    def build_path(self,
        nodes_k: np.ndarray = None,
        nodes_param: np.ndarray = None, # (N_p_nodes, dim_param)
        dim_param = None,
        n_interp: int = 1
    ):
        """
        Build a k-path in the Brillouin zone.

        The `nodes_k` array must have the following shape:
            - shape ``(N_k_nodes, dim_k)`` for any k-path.

        The `nodes_param` array must have the following shape:
            - shape ``(N_p_nodes, dim_param)`` for any parameter path.

        Generally, the path may be mixed, and the resulting mesh will have a combined shape
            - shape ``(N_k_nodes*n_interp, dim_k+dim_param)`` for any path.

        Parameters
        ----------
        nodes_k : np.ndarray
            The k-path points in reduced coordinates.
        nodes_param : np.ndarray
            The parameter path points in reduced coordinates.
        dim_param : int
            The dimension of the parameter space.
        n_interp : int
            The number of interpolation points between each pair of nodes.
        """
        dims = []
        
        if nodes_k is not None:
            if np.asarray(nodes_k).shape[1] != self._dim_k:
                raise ValueError(f"Expected k-space dimension {self._dim_k}, got {np.asarray(nodes_k).shape[1]}")

            k_flat = _interpolate_path(np.asarray(nodes_k), n_interp)
            dims.append(k_flat.shape[0])
        else:
            k_flat = np.zeros((1, 0))

        if nodes_param is not None:
            if dim_param is None:
                raise ValueError("dim_param must be specified if nodes_param is given")
            if np.asarray(nodes_param).shape[1] != dim_param:
                raise ValueError(f"Expected parameter-space dimension {dim_param}, got {np.asarray(nodes_param).shape[1]}")

            self._dim_param = dim_param
            p_flat = _interpolate_path(np.asarray(nodes_param), n_interp)
            dims.append(p_flat.shape[0])
        else:
            self._dim_param = 0
            p_flat = np.zeros((1, 0))

        Nk, Np = k_flat.shape[0], p_flat.shape[0]
        n_k_axes = (1 if nodes_k is not None else 0)
        n_p_axes = (1 if nodes_param is not None else 0)

        # flattened mesh points (N_points, dim_total)
        k_rep = np.repeat(k_flat, Np, axis=0)
        p_rep = np.tile(p_flat, (Nk, 1))
        flat_mesh = np.hstack([k_rep, p_rep])

        self._points = self.path = flat_mesh

        self.mesh_type = "path"
        self.is_grid = False
        self.axis_types = ['k'] * n_k_axes + ['param'] * n_p_axes
        self._k_axes     = list(range(n_k_axes))
        self._param_axes = list(range(n_k_axes, n_k_axes + n_p_axes))
        self._shape_k = (k_flat.shape[0],) if n_k_axes > 0 else None
        self._shape_param = (p_flat.shape[0],) if n_p_axes > 0 else None


    def build_grid(self,
        points: np.ndarray = None,
        shape_k: tuple = None,
        shape_param: tuple = None,
        full_grid: bool = False,
        gamma_centered: bool = False,
        k_endpoints: bool = True,
        param_endpoints: bool = True
    ):
        """ Build a regular k-space and parameter space grid.

        The grid is a set of points in reduced units that form a cubic/square
        lattice in k-space and parameter space. The exact nature of the grid
        (e.g., the number of points, the spacing) is determined by the input
        parameters.

        .. warning::
            You must pass *either* ``full_grid=True``, or the ``points`` array.

        After calling this function, the ``.grid`` attribute will be:
            - shape ``(*shape_k, *shape_param, dim_k+dim_param)`` for full-grid,
        while the ``.flat`` attribute will be the flattened version:
            - shape ``(N_k*N_p, dim_k+dim_param)``.

        Parameters
        ----------
        points : np.ndarray
            The points in k-space and parameter space. The shape should be 
            ``(*shape_k, *shape_param, dim_k + dim_param)``
        shape_k : list or tuple of int with length dim_k
            The shape of the k-space grid.
        shape_param : list or tuple of int with length dim_param, optional
            The shape of the parameter space grid.
        full_grid : bool, optional
            If True, build a full grid in k-space and parameter space.
        gamma_centered : bool, optional
            If True, center the k-space grid at the Gamma point. This
            makes the grid axes go from -0.5 to 0.5.
        exclude_k_endpoints : bool, optional
            If True, exclude the endpoints of the k-space grid.
        exclude_param_endpoints : bool, optional
            If True, exclude the endpoints of the parameter space grid.
        """
        model = self.model
        self.mesh_type = "grid"
        self.is_grid = True

        if full_grid and points is not None:
            raise ValueError("Cannot specify both 'full_grid=True' and 'points'.")
        elif points is None and not full_grid:
            raise ValueError("Must either specify 'points' or 'full_grid=True'.")

        if shape_k is not None:
            if len(shape_k) != self.dim_k:
                raise ValueError(f"Expected k-space dimension {self.dim_k}, got {len(shape_k)}")

            if full_grid:
                k_flat = self.gen_hyper_cube(
                    *shape_k,
                    centered=gamma_centered,
                    flat=True,
                    endpoint=k_endpoints
                )
            else:
                if points.shape[-1] != model.dim_k:
                    raise ValueError(f"Expected k-space dimension {model.dim_k}, got {points.shape[-1]}")
                if points.shape[:-1] != shape_k:
                    raise ValueError(f"Expected k-space shape {shape_k}, got {points.shape[:-1]}")
                k_flat = points.reshape(-1, model.dim_k)
        else:
            k_flat = np.zeros((1, 0))

        if shape_param is not None:
            dim_param = len(shape_param)
            self._dim_param = dim_param

            if full_grid:
                p_flat = self.gen_hyper_cube(
                    *shape_param,
                    centered=gamma_centered,
                    flat=True,
                    endpoint=param_endpoints
                )
            else:
                if points.shape[-1] != self.dim_param:
                    raise ValueError(f"Expected parameter-space dimension {self.dim_param}, got {points.shape[-1]}")
                if points.shape[:-1] != shape_param:
                    raise ValueError(f"Expected parameter-space shape {shape_param}, got {points.shape[:-1]}")
                p_flat = points.reshape(-1, self.dim_param)
        else:
            self._dim_param = 0
            p_flat = np.zeros((1, 0))

        # cross product of k_flat and p_flat
        Nk, Np = k_flat.shape[0], p_flat.shape[0]

        # flattened mesh points (N_points, dim_total)
        k_rep = np.repeat(k_flat, Np, axis=0)
        p_rep = np.tile(p_flat, (Nk, 1))
        flat_mesh = np.hstack([k_rep, p_rep])

        self._points = flat_mesh

        # label each structured axis as k-space or parameter-space
        n_k_axes = len(shape_k) if shape_k is not None else 0
        n_p_axes = len(shape_param) if shape_param is not None else 0

        self.axis_types = ['k'] * n_k_axes + ['param'] * n_p_axes
        self._k_axes     = list(range(n_k_axes))
        self._param_axes = list(range(n_k_axes, n_k_axes + n_p_axes))
        self._shape_k = shape_k
        self._shape_param = shape_param

        # --- precompute k-space phases if present ---
        if self.dim_k > 0:
            pass
            # self.recip_lat_vecs = model.get_recip_lat()
            # self.orb_phases = self.get_orb_phases()
            # self.bc_phase = self.get_boundary_phase()

    def build_custom(self, points, axis_types):
        """Build a custom mesh from the given points and axis types.

        Parameters
        ----------
        points : np.ndarray
            Array of shape (N1, N2, ..., Nd, dim_total) defining the mesh points.
        axis_types : list[str]
            List of axis types ('k' or 'param') corresponding to each axis in the mesh.

        Returns
        -------
        Mesh
            A Mesh object representing the custom mesh.

        Raises
        ------
        ValueError
            If the shape of points or the length of axis_types is inconsistent.
        """
        if not isinstance(points, np.ndarray):
            raise ValueError("Mesh points must be a numpy array.")
        if points.ndim != len(axis_types) + 1:
            raise ValueError("Inconsistent dimensions between mesh points and axis types.")

        self._points = np.reshape(points, (-1, points.shape[-1]))
        self.axis_types = axis_types
        self._k_axes = [i for i, at in enumerate(axis_types) if at == 'k']
        self._param_axes = [i for i, at in enumerate(axis_types) if at == 'param']
        self._dim = points.shape[-1]
        self._dim_k = len(self._k_axes)
        self._dim_param = len(self._param_axes)
        self._shape = points.shape[:-1]
        self._shape_k = tuple(self._shape[i] for i in self._k_axes)
        self._shape_param = tuple(self._shape[i] for i in self._param_axes)

        return

    @staticmethod
    def gen_hyper_cube(
        *n_points, centered: bool = False, flat: bool = True, endpoint: bool = False
    ) -> np.ndarray:
        """Generate a hypercube of points in the specified dimensions.

        Parameters
        ----------
        *n_points: int
            Number of points along each dimension.
    
        centered: bool, optional
            If True, mesh is defined from [-0.5, 0.5] along each direction.
            Defaults to False.
        flat: bool, optional
            If True, returns flattened array of k-points (e.g. of shape ``(n1*n2*n3 , 3)``).
            If False, returns reshaped array with axes along each k-space dimension
            (e.g. of shape ``(1, n1, n2, n3, 3)``). Defaults to True.
        endpoint: bool, optional
            If True, includes 1 (edge of BZ in reduced coordinates) in the mesh. Defaults to False.
            When Wannierizing should omit this point.

        Returns
        -------
        mesh: np.ndarray
            Array of coordinates defining the hypercube. 
        """

        end_pts = [-0.5, 0.5] if centered else [0, 1]
        vals = [
            np.linspace(end_pts[0], end_pts[1], n, endpoint=endpoint)
            for n in n_points
        ]
        flat_mesh = np.stack(np.meshgrid(*vals, indexing="ij"), axis=-1)

        return flat_mesh if not flat else flat_mesh.reshape(-1, len(vals))
    

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
        r"""Returns exp(\pm i k.tau) factors

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