import numpy as np

def k_uniform_mesh(model, mesh_size) -> np.ndarray:
    r"""
    Returns a uniform grid of k-points that can be passed to
    passed to function :func:`pythtb.tb_model.solve_all`.  This
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
    if use_mesh.shape != (model._dim_k,):
        print(use_mesh.shape)
        raise Exception("\n\nIncorrect size of the specified k-mesh!")
    if np.min(use_mesh) <= 0:
        raise Exception("\n\nMesh must have positive non-zero number of elements.")

    axes = [np.linspace(0, 1, n, endpoint=False) for n in use_mesh]
    mesh = np.meshgrid(*axes, indexing='ij')
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
    dim = model._dim_k

    # Parse kpts and validate
    k_list = _parse_kpts(kpts, dim)
    if k_list.shape[1] != dim:
        raise ValueError(f"Dimension mismatch: kpts shape {k_list.shape}, model dim {dim}")
    if nk < len(k_list):
        raise ValueError("nk must be >= number of nodes in kpts")

    # Extract periodic lattice and compute k-space metric
    lat_per = model._lat[model._per]
    k_metric = np.linalg.inv(lat_per @ lat_per.T)

    # Compute segment vectors and lengths in Cartesian metric
    diffs = k_list[1:] - k_list[:-1]
    seg_lengths = np.sqrt(np.einsum('ij,ij->i', diffs @ k_metric, diffs))

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
            'full': [[0.0], [0.5], [1.0]],
            'fullc': [[-0.5], [0.0], [0.5]],
            'half': [[0.0], [0.5]],
        }
        return np.array(presets[kpts], float)

    arr = np.array(kpts, float)
    if arr.ndim == 1 and dim == 1:
        arr = arr[:, None]
    return arr

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
        length = k_node[n] - k_node[n-1]
        print(
            f"  Node {n-1} {k_list[n-1]} to Node {n} {k_list[n]}: "
            f"distance = {length:.5f}"
        )

    print("Node distances (cumulative):", k_node)
    print("Node indices in path:", node_index)
    print("-------------------------")