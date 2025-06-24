import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
import matplotlib.colors as mcolors
from .utils import pauli_decompose


def _fmt_num(x, precision=3):
    # If the imaginary part is negligible, print as a real number.
    if abs(x.imag) < 1e-10:
        if x.real == 1:
            return ""
        elif x.real == -1:
            return "-"
        else:
            return f"{x.real:.{precision}g}"
    elif abs(x.real) < 1e-10:
        if x.imag == 1:
            return "i"
        elif x.imag == -1:
            return "-i"
        else:
            return f"{x.imag:.{precision}g}i"
    else:
        return f"({x:.{precision}g})"


def _pauli_decompose_str(M, precision=3):

    a0, a1, a2, a3 = pauli_decompose(M)

    # Build a list of terms, including only those with non-negligible coefficients.
    terms = []
    if abs(a0) > 1e-10:
        terms.append(f"{_fmt_num(a0, precision=precision)} \sigma_0")
    if abs(a1) > 1e-10:
        terms.append(f"{_fmt_num(a1, precision=precision)} \sigma_x")
    if abs(a2) > 1e-10:
        terms.append(f"{_fmt_num(a2, precision=precision)} \sigma_y")
    if abs(a3) > 1e-10:
        terms.append(f"{_fmt_num(a3, precision=precision)} \sigma_z")

    # If all coefficients are zero, return "0".
    if not terms:
        return "0"

    return " + ".join(terms).replace("+ -", "- ")


def _pauli_decompose_unicode(M, precision=3):
    """
    Decompose a 2x2 matrix M in terms of the Pauli matrices and return
    a Unicode string representation.

    That is, find coefficients a0, a1, a2, a3 such that:

        M = a0 * I + a1 * σₓ + a2 * σᵧ + a3 * σᶻ

    Parameters:
        M (array-like): A 2x2 matrix.
        precision (int): Number of significant digits for the coefficients.

    Returns:
        str: A Unicode string representing the decomposition.
    """
    # Use your existing function to get the coefficients
    a0, a1, a2, a3 = pauli_decompose(M)

    def fmt(x):
        # If the imaginary part is negligible, print as a real number.
        if abs(x.imag) < 1e-10:
            if x.real == 1:
                return ""
            elif x.real == -1:
                return "-"
            else:
                return f"{x.real:.{precision}g}"
        elif abs(x.real) < 1e-10:
            if x.imag == 1:
                return "i"
            elif x.imag == -1:
                return "-i"
            else:
                return f"{x.imag:.{precision}g}i"
        else:
            return f"({x:.{precision}g})"

    # Build a list of terms, using Unicode symbols.
    terms = []
    if abs(a0) > 1e-10:
        terms.append(f"{fmt(a0)} 𝟙")
    if abs(a1) > 1e-10:
        # Using Unicode subscript x (ₓ, U+2093)
        terms.append(f"{fmt(a1)} σ_x")
    if abs(a2) > 1e-10:
        # For y, using a common Unicode modifier letter for small y: ᵧ (U+1D67)
        terms.append(f"{fmt(a2)} σ_y")
    if abs(a3) > 1e-10:
        # For z, using a Unicode modifier letter small z: ᶻ (U+1D5B)
        terms.append(f"{fmt(a3)} σ_z")

    if not terms:
        return "0"

    return " + ".join(terms).replace("+ -", "- ")


# TODO: Add hoverable hopping and onsite terms
def plot_tb_model(
    model,
    proj_plane=None,
    eig_dr=None,
    draw_hoppings=True,
    annotate_onsite_en=False,
    ph_color="black",
    orb_color="r",
):
    r"""

    Rudimentary function for visualizing tight-binding model geometry,
    hopping between tight-binding orbitals, and electron eigenstates.

    If eigenvector is not drawn, then orbitals in home cell are drawn
    as red circles, and those in neighboring cells are drawn with
    different shade of red. Hopping term directions are drawn with
    green lines connecting two orbitals. Origin of unit cell is
    indicated with blue dot, while real space unit vectors are drawn
    with blue lines.

    If eigenvector is drawn, then electron eigenstate on each orbital
    is drawn with a circle whose size is proportional to wavefunction
    amplitude while its color depends on the phase. There are various
    coloring schemes for the phase factor; see more details under
    *ph_color* parameter. If eigenvector is drawn and coloring scheme
    is "red-blue" or "wheel", all other elements of the picture are
    drawn in gray or black.

    :param dir_first: First index of Cartesian coordinates used for
        plotting.

    :param dir_second: Second index of Cartesian coordinates used for
        plotting. For example if dir_first=0 and dir_second=2, and
        Cartesian coordinates of some orbital is [2.0,4.0,6.0] then it
        will be drawn at coordinate [2.0,6.0]. If dimensionality of real
        space (*dim_r*) is zero or one then dir_second should not be
        specified.

    :param eig_dr: Optional parameter specifying eigenstate to
        plot. If specified, this should be one-dimensional array of
        complex numbers specifying wavefunction at each orbital in
        the tight-binding basis. If not specified, eigenstate is not
        drawn.

    :param draw_hoppings: Optional parameter specifying whether to
        draw all allowed hopping terms in the tight-binding
        model. Default value is True.

    :param ph_color: Optional parameter determining the way
        eigenvector phase factors are translated into color. Default
        value is "black". Convention of the wavefunction phase is as
        in convention 1 in section 3.1 of :download:`notes on
        tight-binding formalism  <misc/pythtb-formalism.pdf>`.  In
        other words, these wavefunction phases are in correspondence
        with cell-periodic functions :math:`u_{n {\bf k}} ({\bf r})`
        not :math:`\Psi_{n {\bf k}} ({\bf r})`.

        * "black" -- phase of eigenvectors are ignored and wavefunction
        is always colored in black.

        * "red-blue" -- zero phase is drawn red, while phases or pi or
        -pi are drawn blue. Phases in between are interpolated between
        red and blue. Some phase information is lost in this coloring
        becase phase of +phi and -phi have same color.

        * "wheel" -- each phase is given unique color. In steps of pi/3
        starting from 0, colors are assigned (in increasing hue) as:
        red, yellow, green, cyan, blue, magenta, red.

    :returns:
        * **fig** -- Figure object from matplotlib.pyplot module
        that can be used to save the figure in PDF, EPS or similar
        format, for example using fig.savefig("name.pdf") command.
        * **ax** -- Axes object from matplotlib.pyplot module that can be
        used to tweak the plot, for example by adding a plot title
        ax.set_title("Title goes here").

    Example usage::

        # Draws x-y projection of tight-binding model
        # tweaks figure and saves it as a PDF.
        (fig, ax) = tb.visualize(0, 1)
        ax.set_title("Title goes here")
        fig.savefig("model.pdf")

    See also these examples: :ref:`edge-example`,
    :ref:`visualize-example`.

    """

    cmap = plt.get_cmap("hsv", model._norb)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Projection function: projects a vector onto the 2D plane
    def proj(v):
        if proj_plane is not None:
            coord_x = v[proj_plane[0]]
            coord_y = v[proj_plane[1]]
        else:
            coord_x = v[0]
            coord_y = v[1]
        return [coord_x, coord_y]

    # Convert reduced coordinates to Cartesian coordinates
    def to_cart(red):
        return np.dot(red, model._lat)

    # to ensure proper padding, track all plotted coordinates
    all_coords = []

    # Draw the origin
    origin = [0.0, 0.0]
    ax.plot(origin[0], origin[1], "X", color="black", ms=8)
    all_coords.append(origin)

    # Draw lattice (unit cell) vectors as arrows and label them
    # TODO: Fix lattice vectors to appear in hybrid finite/periodic models
    ends = []
    for i in model._per:
        start = origin
        end = proj(model._lat[i])
        ends.append(end)

        # lattice vector arrow
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=15,
            color="blue",
            lw=2,
            zorder=0,
        )
        ax.add_patch(arrow)

        # annotation of lattice
        ax.annotate(
            f"$\\vec{{a}}_{i}$",
            xy=end,  # (end[0], end[1])
            xytext=(4, 4),  # offset in points
            textcoords="offset points",
            color="blue",
            fontsize=12,
            va="bottom",
            ha="right",
        )
        all_coords.append(end)

    # plot dotted bounding lines to unit cell
    ends = np.array(ends)

    # if 2d cell
    if ends.shape[0] > 1:
        # top shifted line
        start = ends[0]
        end = ends[0] + ends[1]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            ls="--",
            lw=1,
            color="b",
            zorder=0,
            alpha=0.5,
        )

        # right shifted line
        start = ends[1]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            ls="--",
            lw=1,
            color="b",
            zorder=0,
            alpha=0.5,
        )

    # Draw orbitals: home-cell orbitals in red
    orb_coords = []
    for i in range(model._norb):
        pos = to_cart(model._orb[i])
        p = proj(pos)
        color = cmap(i)
        ax.scatter(p[0], p[1], color=orb_color, s=50, zorder=2, label=f"Orbital {i}")
        orb_coords.append(p)

        # For spinful case, annotate orbital with onsite decomposition.
        if model._nspin == 2 and annotate_onsite_en:
            onsite_str = _pauli_decompose_str(model._site_energies[i])
            ax.annotate(
                f"$\Delta_{{{i}}} = {onsite_str}$",
                xy=p,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                color="red",
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="lightcoral", ec="none", alpha=0.6
                ),
                zorder=5,
            )
        elif model._nspin == 1 and annotate_onsite_en:
            onsite_str = f"$\Delta_{{{i}}} = {model._site_energies[i]:.2f}$"
            ax.annotate(
                onsite_str,
                xy=p,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                color="red",
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="lightcoral", ec="none", alpha=0.6
                ),
                zorder=5,
            )

    # Draw hopping terms with curved arrows
    hopping_coords = []

    # maximum magnitudes of hopping strengths
    mags = [np.amax(abs(hop[0])) for hop in model._hoppings]
    biggest_hop = np.amax(mags)
    # transparency propto hopping strength
    arrow_alphas = mags / biggest_hop if biggest_hop != 0 else np.ones(len(mags))
    arrow_alphas = (
        0.3 + 0.7 * arrow_alphas**2
    )  # nonlinear mapping for greater visual difference

    if draw_hoppings:
        for h_idx, h in enumerate(model._hoppings):
            amp = h[0]
            i_orb = h[1]
            j_orb = h[2]

            r_vec = None
            intracell = True
            if model._dim_k != 0 and len(h) == 4:
                r_vec = h[3]
                intracell = np.all(r_vec == 0)

            for shift in range(2):  # draw both i->j+R and i-R->j hop
                pos_i = to_cart(model._orb[i_orb])
                pos_j = to_cart(model._orb[j_orb])

                # Determine starting and ending orbital positions
                if r_vec is not None:
                    # Adjust pos_j with lattice translation if provided
                    if shift == 0:
                        # i->j+R
                        pos_j += np.dot(r_vec, model._lat)
                    elif shift == 1:
                        # i-R->j
                        pos_i -= np.dot(r_vec, model._lat)

                p_i = proj(pos_i)
                p_j = proj(pos_j)

                # plot neighboring cell orbital
                # ensure we don't plot orbital in unit cell again (if no translation)
                if not intracell:
                    # ensure we only scatter orbitals once
                    if p_j not in hopping_coords and shift == 0:
                        ax.scatter(
                            p_j[0],
                            p_j[1],
                            color=orb_color,
                            s=50,
                            zorder=1,
                            alpha=0.5,
                        )
                    if p_i not in hopping_coords and shift == 1:
                        ax.scatter(
                            p_i[0],
                            p_i[1],
                            color=orb_color,
                            s=50,
                            zorder=1,
                            alpha=0.5,
                        )

                # Don't want to plot connecting arrows within unit cell twice
                if intracell and shift == 1:
                    # Arrow connects orbitals within cell, so shift = 1 is same
                    # conditions as shift = 2 (same arrows)
                    continue

                # First arrow: p_i -> p_j
                arrow1 = FancyArrowPatch(
                    p_i,
                    p_j,
                    connectionstyle="arc3,rad=0.08",
                    arrowstyle="->",
                    mutation_scale=15,
                    color="green",
                    lw=1.3,
                    alpha=arrow_alphas[h_idx],
                    zorder=1,
                )
                ax.add_patch(arrow1)

                # Second arrow: p_j -> p_i
                arrow2 = FancyArrowPatch(
                    p_j,
                    p_i,
                    connectionstyle="arc3,rad=0.08",
                    arrowstyle="->",
                    mutation_scale=15,
                    color="green",
                    lw=1.3,
                    alpha=arrow_alphas[h_idx],
                    zorder=1,
                )
                ax.add_patch(arrow2)

                hopping_coords.append(p_i)
                hopping_coords.append(p_j)

    # If eigenstate is provided, overlay eigenstate information on the orbitals
    if eig_dr is not None:
        # For each orbital, size the marker by amplitude and color by phase
        cmap = cm.hsv
        for i in range(model._norb):
            pos = to_cart(model._orb[i])
            p = proj(pos)
            amp = (
                np.abs(eig_dr[i]) ** 2
            )  # intensity proportional to probability density
            phase = np.angle(eig_dr[i])
            # Map phase from [-pi, pi] to [0,1]
            color = cmap((phase + np.pi) / (2 * np.pi))
            if ph_color == "black":
                color = "k"
            ax.scatter(
                p[0],
                p[1],
                s=30 * amp * 2 * model._norb,  # size proportional to amplitude
                color=color,
                edgecolor="k",
                zorder=10,
                alpha=0.7,
                label="Eigenstate" if i == 0 else None,
            )

    # Adjust the axis so everything fits
    all_coords += hopping_coords
    all_coords += orb_coords

    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Add some padding
    pad_x = 0.1 * (max_x - min_x if max_x != min_x else 1)
    pad_y = 0.1 * (max_y - min_y if max_y != min_y else 1)
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)

    # Final plot adjustments
    ax.set_aspect("equal")
    if proj_plane is not None:
        ax.set_xlabel(f"Cartesian coordinate {proj_plane[0]}")
        ax.set_ylabel(f"Cartesian coordinate {proj_plane[1]}")
    else:
        ax.set_xlabel(f"Cartesian coordinate {0}")
        ax.set_ylabel(f"Cartesian coordinate {1}")

    # ax.legend(loc="upper right", fontsize=10)
    # plt.tight_layout()

    return fig, ax


def plot_tb_model_3d(
    model,
    eig_dr=None,
    draw_hoppings=True,
    show_model_info=True,
    site_colors=None,
    site_names=None,
    ph_color="black",
):
    r"""
    Visualize a 3D tight-binding model using Plotly.

    This function creates an interactive 3D plot of your tight-binding model,
    showing the unit-cell origin, lattice vectors (with arrowheads), orbitals,
    hopping lines, and (optionally) an eigenstate overlay with marker sizes
    proportional to amplitude and colors reflecting the phase.

    :param eig_dr: Optional eigenstate (1D array of complex numbers) to display.
    :param draw_hoppings: Whether to draw hopping lines between orbitals.
    :param annotate_onsite_en: Whether to annotate orbitals with onsite energies.
    :param ph_color: Coloring scheme for eigenstate phases (e.g. "black", "red-blue", "wheel").

    :returns:
    * **fig** -- A Plotly Figure object.
    """
    import plotly.graph_objects as go

    # Helper: Convert reduced coordinates to Cartesian coordinates.
    def to_cart(red):
        return np.dot(red, model._lat)

    # Container for all Plotly traces.
    traces = []
    all_coords = []

    # --- Draw Origin ---
    origin = np.array([0.0, 0.0, 0.0])

    all_coords.append(origin)

    # --- Draw Lattice Vectors ---
    # We assume model._per is an iterable of indices for lattice vectors.
    lattice_traces = []
    for i in model._per:
        start = origin
        end = np.array(model._lat[i])
        # Line for the lattice vector.
        lattice_traces.append(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color="blue", width=4),
                showlegend=False,
                hoverinfo="none",
            )
        )
        # Add a cone to simulate an arrowhead.
        direction = end - start
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction_unit = direction / norm
            lattice_traces.append(
                go.Cone(
                    x=[end[0]],
                    y=[end[1]],
                    z=[end[2]],
                    u=[direction_unit[0]],
                    v=[direction_unit[1]],
                    w=[direction_unit[2]],
                    anchor="tip",
                    sizemode="absolute",
                    sizeref=0.2,
                    showscale=False,
                    colorscale=[[0, "blue"], [1, "blue"]],
                    name=f"a{i}",
                )
            )
        # Add a text annotation (using a text scatter) at the end.
        lattice_traces.append(
            go.Scatter3d(
                x=[end[0]],
                y=[end[1]],
                z=[end[2]],
                mode="text",
                # text=[fr"$\vec{{a}}_{i}$"],
                text=[f"a{i}"],
                textposition="top center",
                textfont=dict(color="blue", size=12),
                showlegend=False,
                hoverinfo="none",
            )
        )
        all_coords.append(end)
    traces.extend(lattice_traces)

    # --- Draw Orbitals ---
    orb_x, orb_y, orb_z = [], [], []
    orb_text = []
    orb_marker_colors = []
    onsite_labels = []
    cmap_orb = cm.get_cmap("viridis", model._norb)
    for i in range(model._norb):
        orb_text.append(f"Orbital {i}")

        if model._nspin == 2:
            onsite_str = _pauli_decompose_unicode(model._site_energies[i])
            onsite_label = rf"{onsite_str}"
        else:
            onsite_label = rf"{model._site_energies[i]:.2f}"
        onsite_labels.append(onsite_label)

        pos = to_cart(model._orb[i])
        orb_x.append(pos[0])
        orb_y.append(pos[1])
        orb_z.append(pos[2])
        all_coords.append(pos)

        if site_colors is not None:
            # Use provided color for orbitals.
            orb_marker_colors.append(site_colors[i])

        else:
            # Convert RGBA to hex.
            orb_marker_colors.append(mcolors.to_hex(cmap_orb(i)))

        if site_names is not None:
            name = site_names[i]
        else:
            name = f"Orbital {i}"

        traces.append(
            go.Scatter3d(
                x=[pos[0]],
                y=[pos[1]],
                z=[pos[2]],
                mode="markers",
                marker=dict(color=orb_marker_colors[i], size=10),
                text=[rf"Orbital {i}, Onsite Energy = {onsite_label}"],
                hoverinfo="text",
                name=name,
            )
        )

    # Draw hopping terms ---
    if draw_hoppings:
        hopping_traces = []
        # Compute hopping strengths for opacity.
        mags = [np.amax(np.abs(hop[0])) for hop in model._hoppings]
        biggest_hop = np.amax(mags) if mags else 1.0
        arrow_alphas = (
            np.array(mags) / biggest_hop if biggest_hop != 0 else np.ones(len(mags))
        )
        arrow_alphas = 0.3 + 0.7 * arrow_alphas**2  # Non-linear mapping.
        for h_idx, h in enumerate(model._hoppings):
            amp = h[0]
            i_orb = h[1]
            j_orb = h[2]
            r_vec = None
            intracell = True

            if model._dim_k != 0 and len(h) == 4:
                r_vec = h[3]
                intracell = np.all(np.array(r_vec) == 0)

            # Draw hopping for both directions.
            for shift in range(2):
                pos_i = to_cart(model._orb[i_orb])
                pos_j = to_cart(model._orb[j_orb])
                if r_vec is not None:
                    if shift == 0:
                        pos_j = pos_j + np.dot(r_vec, model._lat)
                    elif shift == 1:
                        pos_i = pos_i - np.dot(r_vec, model._lat)

                if not intracell:
                    # ensure we only scatter orbitals once
                    traces.append(
                        go.Scatter3d(
                            x=[pos_i[0]],
                            y=[pos_i[1]],
                            z=[pos_i[2]],
                            mode="markers",
                            marker=dict(color=orb_marker_colors[i_orb], size=8),
                            name="",
                            showlegend=False,
                            text=[
                                f"Orbital {i_orb}, \n Onsite Energy: {onsite_labels[i_orb]}"
                            ],
                            hoverinfo="text",
                        )
                    )

                    traces.append(
                        go.Scatter3d(
                            x=[pos_j[0]],
                            y=[pos_j[1]],
                            z=[pos_j[2]],
                            mode="markers",
                            marker=dict(color=orb_marker_colors[j_orb], size=8),
                            name="",
                            showlegend=False,
                            text=[
                                rf"Orbital {j_orb}, onsite energy: {onsite_labels[j_orb]}"
                            ],
                            hoverinfo="text",
                        )
                    )

                # Don't want to plot connecting arrows within unit cell twice
                if intracell and shift == 1:
                    # Arrow connects orbitals to home cell, so shift = 1 is same
                    # conditions as shift = 2 (same arrows)
                    continue

                if model._nspin == 2:
                    amp_str = _pauli_decompose_unicode(amp)
                else:
                    amp_str = f"{amp:.2f}"

                if r_vec is not None:
                    r_vec_str = np.array2string(r_vec, precision=2, separator=", ")
                    r_vec_str = r_vec_str.replace("\n", "")
                    r_vec_str = r_vec_str.replace(" ", "")
                    hop_str = f"Hopping {i_orb} --> {j_orb} + {r_vec} = {amp_str}"
                else:
                    hop_str = f"Hopping {i_orb} --> {j_orb} = {amp_str}"

                hopping_traces.append(
                    go.Scatter3d(
                        x=[pos_i[0], pos_j[0]],
                        y=[pos_i[1], pos_j[1]],
                        z=[pos_i[2], pos_j[2]],
                        mode="lines",
                        line=dict(
                            color="green",
                            width=3,
                        ),
                        opacity=arrow_alphas[h_idx],
                        text=hop_str,
                        showlegend=False,
                        hoverinfo="text",
                    )
                )
                all_coords.append(pos_i)
                all_coords.append(pos_j)
        traces.extend(hopping_traces)

    # Overlay eigenstate if provided
    if eig_dr is not None:
        eigen_x, eigen_y, eigen_z = [], [], []
        eigen_marker_sizes = []
        eigen_marker_colors = []
        cmap_phase = cm.get_cmap("hsv")
        for i in range(model._norb):
            pos = to_cart(model._orb[i])
            eigen_x.append(pos[0])
            eigen_y.append(pos[1])
            eigen_z.append(pos[2])
            amp = np.abs(eig_dr[i]) ** 2  # amplitude propto probability density.
            eigen_marker_sizes.append(10 * amp)  # Scale factor for magnitude.
            phase = np.angle(eig_dr[i])
            eigen_marker_colors.append(
                mcolors.to_hex(cmap_phase((phase + np.pi) / (2 * np.pi)))
            )
        traces.append(
            go.Scatter3d(
                x=eigen_x,
                y=eigen_y,
                z=eigen_z,
                mode="markers",
                marker=dict(
                    color=eigen_marker_colors,
                    size=eigen_marker_sizes,
                    symbol="circle",
                    line=dict(color="black", width=1),
                ),
                name="Eigenstate",
            )
        )

    # --- Determine Axis Limits ---
    all_coords = np.array(all_coords)
    min_x, max_x = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
    min_y, max_y = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
    min_z, max_z = np.min(all_coords[:, 2]), np.max(all_coords[:, 2])
    pad_x = 0.1 * (max_x - min_x if max_x != min_x else 1)
    pad_y = 0.1 * (max_y - min_y if max_y != min_y else 1)
    pad_z = 0.1 * (max_z - min_z if max_z != min_z else 1)

    layout = go.Layout(
        scene=dict(
            xaxis=dict(range=[min_x - pad_x, max_x + pad_x], title="X"),
            yaxis=dict(range=[min_y - pad_y, max_y + pad_y], title="Y"),
            zaxis=dict(range=[min_z - pad_z, max_z + pad_z], title="Z"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    fig = go.Figure(data=traces, layout=layout)

    def get_pretty_model_info_str():
        lines = []
        lines.append("<b>Tight-Binding Model Information</b><br>")
        lines.append("<br>")
        lines.append("<b>Lattice Vectors:</b><br>")
        for i, vec in enumerate(model._lat):
            lines.append(
                f"a_{i} = {np.array2string(vec, precision=3, separator=', ')}<br>"
            )
        lines.append("<br>")
        lines.append("<b>Orbital Vectors:</b><br>")
        for i, orb in enumerate(model._orb):
            lines.append(
                f"Orbital {i} = {np.array2string(orb, precision=3, separator=', ')}<br>"
            )
        lines.append("<br>")
        lines.append(f"<b>Number of Spins:</b> {model._nspin}")
        return "".join(lines)

    report_text = get_pretty_model_info_str()

    if show_model_info:
        # 3) Add an annotation. We’ll place it in the upper-left corner (x=0.01, y=0.99).
        fig.add_annotation(
            text=report_text,
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            showarrow=False,
            align="left",
            font=dict(family="Courier New, monospace", size=12, color="black"),
            bordercolor="black",
            borderwidth=1,
            borderpad=5,
            bgcolor="white",
        )

    return fig


def plot_bands(
    model,
    k_path,
    nk=101,
    evals=None, 
    evecs=None,
    k_label=None,
    proj_orb_idx=None,
    proj_spin=False,
    fig=None,
    ax=None,
    title=None,
    scat_size=3,
    lw=2,
    lc="b",
    ls="solid",
    cmap="plasma",
    show=False,
    cbar=True,
):
    """

    Args:
        k_path (list): List of high symmetry points to plot bands through
        k_label (list[str], optional): Labels of high symmetry points. Defaults to None.
        title (str, optional): _description_. Defaults to None.
        save_name (str, optional): _description_. Defaults to None.
        red_lat_idx (list, optional): _description_. Defaults to None.
        show (bool, optional): _description_. Defaults to False.

    Returns:
        fig, ax: matplotlib fig and ax
    """

    if fig is None:
        fig, ax = plt.subplots()

    # generate k-path and labels
    (k_vec, k_dist, k_node) = model.k_path(k_path, nk, report=False)

    # scattered bands with sublattice color
    if proj_orb_idx is not None:
        if evals is None or evecs is None:
            # diagonalize model on path
            evals, evecs = model.solve_ham(k_vec, return_eigvecs=True)
        n_eigs = evals.shape[-1]
        wt = abs(evecs) ** 2

        if model._nspin == 1:
            col = np.sum([wt[..., i] for i in proj_orb_idx], axis=0)
        elif model._nspin == 2:
            col = np.sum([wt[..., i, :] for i in proj_orb_idx], axis=(0, -1))

        for n in range(n_eigs):
            scat = ax.scatter(
                k_dist,
                evals[:, n],
                c=col[:, n],
                cmap=cmap,
                marker="o",
                s=scat_size,
                vmin=0,
                vmax=1,
                zorder=2,
            )

        if cbar:
            cbar = fig.colorbar(scat, ticks=[1, 0], pad=0.01)
            # cbar.set_ticks([])
            # cbar.ax.set_yticklabels([r'$B$', r'$A$'], size=12)
            cbar.ax.set_yticklabels(
                [
                    r"$ |\langle \psi_{nk} | \phi_B \rangle |^2$",
                    r"$|\langle \psi_{nk} | \phi_A \rangle |^2$",
                ],
                size=12,
            )

    elif proj_spin:
        if evals is None or evecs is None:
            # diagonalize model on path
            evals, evecs = model.solve_ham(k_vec, return_eigvecs=True)
        n_eigs = evals.shape[-1]

        if model._nspin <= 1:
            raise ValueError("Spin needs to be greater than 1 for projecting spin.")

        wt = abs(evecs) ** 2
        col = np.sum(wt[..., 1], axis=2)

        for n in range(n_eigs):
            scat = ax.scatter(
                k_dist,
                evals[:, n],
                c=col[:, n],
                cmap=cmap,
                marker="o",
                s=scat_size,
                vmin=0,
                vmax=1,
                zorder=2,
            )

        cbar = fig.colorbar(scat, ticks=[1, 0])
        cbar.ax.set_yticklabels(
            [
                r"$ |\langle \psi_{nk} | \chi_{\uparrow} \rangle |^2$",
                r"$|\langle \psi_{nk} | \chi_{\downarrow} \rangle |^2$",
            ],
            size=12,
        )

    else:
        if evals is None:
            evals = model.solve_ham(k_vec, return_eigvecs=False)
        n_eigs = evals.shape[-1]

        # continuous bands
        for n in range(n_eigs):
            ax.plot(k_dist, evals[:, n], c=lc, lw=lw, ls=ls)

    ax.set_xlim(0, k_node[-1])
    ax.set_xticks(k_node)
    for n in range(len(k_node)):
        ax.axvline(x=k_node[n], linewidth=0.5, color="k", zorder=1)
    if k_label is not None:
        ax.set_xticklabels(k_label, size=12)

    ax.set_title(title)
    ax.set_ylabel(r"Energy $E(\mathbf{{k}})$", size=12)
    ax.yaxis.labelpad = 10

    if show:
        plt.show()

    return fig, ax
