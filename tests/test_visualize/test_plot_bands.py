import inspect
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pythtb import Lattice, TBModel
from pythtb.visualization.tbmodel import plot_bands


K_NODES = [[0.0], [0.25], [0.5]]
K_LABELS = ["A", "B", "C"]
NK = 9


@pytest.fixture(autouse=True)
def close_figures():
    existing = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - existing:
        plt.close(number)


@pytest.fixture
def model():
    lattice = Lattice([[2.0]], [[0.0], [0.5]], periodic_dirs=[0])
    model = TBModel(lattice, spinful=True)
    model.set_onsite([[0.2, 0.0, 0.0, 0.1], [-0.2, 0.0, 0.0, -0.1]])
    model.set_hop(1.0, 0, 1, [0])
    model.set_hop(0.6, 0, 1, [1])
    return model


@pytest.fixture(
    params=[{}, {"proj_orb_idx": [0]}, {"proj_spin": True}],
    ids=["plain", "orbital", "spin"],
)
def projection(request):
    return request.param


@pytest.mark.parametrize("provided", ["neither", "figure", "axes", "both"])
def test_figure_and_axes(model, projection, provided):
    fig = ax = None
    kwargs = {}
    if provided == "figure":
        fig = plt.figure()
        kwargs["fig"] = fig
    elif provided in ("axes", "both"):
        fig, axes = plt.subplots(1, 2)
        ax = axes[1]
        kwargs["ax"] = ax
        if provided == "both":
            kwargs["fig"] = fig

    result_fig, result_ax = model.plot_bands(
        K_NODES,
        K_LABELS,
        NK,
        bands_label="bands",
        cbar=False,
        **projection,
        **kwargs,
    )

    if fig is not None:
        assert result_fig is fig
    if ax is not None:
        assert result_ax is ax
        assert not axes[0].lines
        assert not axes[0].collections
    assert result_ax.figure is result_fig
    assert len(result_fig.axes) == (2 if ax is not None else 1)

    k_vec, k_dist, node_dist = model.k_path(K_NODES, NK)
    evals = model.solve_ham(k_vec)
    if projection:
        assert len(result_ax.collections) == model.nstate
        for band, collection in enumerate(result_ax.collections):
            np.testing.assert_allclose(
                collection.get_offsets(), np.column_stack((k_dist, evals[:, band]))
            )
    else:
        for band, line in enumerate(result_ax.lines[: model.nstate]):
            np.testing.assert_allclose(line.get_xdata(), k_dist)
            np.testing.assert_allclose(line.get_ydata(), evals[:, band])

    np.testing.assert_allclose(result_ax.get_xticks(), node_dist)
    np.testing.assert_allclose(result_ax.get_xlim(), node_dist[[0, -1]])
    for line, distance in zip(result_ax.lines[-len(node_dist) :], node_dist):
        np.testing.assert_allclose(line.get_xdata(), [distance, distance])
    assert [tick.get_text() for tick in result_ax.get_xticklabels()] == K_LABELS
    assert result_ax.get_legend_handles_labels()[1] == ["bands"]
    assert result_ax.get_legend() is not None


def test_figure_only_reuses_current_axes(model, projection):
    fig, axes = plt.subplots(1, 2)
    fig.sca(axes[0])
    plt.figure()  # A different pyplot figure must not receive the bands.

    result_fig, result_ax = model.plot_bands(
        K_NODES, nk=NK, fig=fig, cbar=False, **projection
    )

    assert result_fig is fig
    assert result_ax is axes[0]
    assert len(fig.axes) == 2


def test_mismatched_figure_and_axes(model, projection):
    fig = plt.figure()
    other_fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="fig.*ax|ax.*fig"):
        model.plot_bands(K_NODES, nk=NK, fig=fig, ax=ax, **projection)

    assert not fig.axes
    assert other_fig.axes == [ax]
    assert not ax.lines
    assert not ax.collections


@pytest.mark.parametrize("cbar", [False, True])
def test_colorbar_flag(model, projection, cbar):
    fig, ax = model.plot_bands(K_NODES, nk=NK, cbar=cbar, **projection)

    assert len(fig.axes) == 1 + int(cbar and bool(projection))
    assert fig.axes[0] is ax
    if cbar and projection:
        np.testing.assert_allclose(fig.axes[1].get_yticks(), [1, 0])


@pytest.mark.parametrize(
    "projection, flatten_spin_axis",
    [
        ({}, True),
        ({"proj_orb_idx": [0]}, True),
        ({"proj_orb_idx": [0]}, False),
        ({"proj_spin": True}, False),
    ],
    ids=["plain", "orbital-flat", "orbital-spin-axis", "spin"],
)
@pytest.mark.parametrize("standalone", [False, True], ids=["method", "standalone"])
def test_precomputed_eigenpairs(
    model, monkeypatch, projection, flatten_spin_axis, standalone
):
    k_vec, _, _ = model.k_path(K_NODES, NK)
    evals, evecs = model.solve_ham(
        k_vec, return_eigvecs=True, flatten_spin_axis=flatten_spin_axis
    )
    evals = evals + 2.0  # Distinguish supplied values from a fresh model solve.

    def unexpected_solve(*args, **kwargs):
        pytest.fail("Precomputed eigenpairs should be used without solving again")

    monkeypatch.setattr(model, "solve_ham", unexpected_solve)
    kwargs = dict(
        k_nodes=K_NODES,
        k_node_labels=K_LABELS,
        nk=NK,
        evals=evals,
        evecs=evecs if projection else None,
        cbar=False,
        **projection,
    )
    if standalone:
        fig, ax = plot_bands(model, **kwargs)
    else:
        fig, ax = model.plot_bands(**kwargs)

    assert ax.figure is fig
    assert [tick.get_text() for tick in ax.get_xticklabels()] == K_LABELS
    if projection:
        weights = abs(evecs.reshape(NK, model.nstate, model.norb, 2)) ** 2
        expected = (
            weights[:, :, 0, :].sum(axis=-1)
            if "proj_orb_idx" in projection
            else weights[..., 1].sum(axis=2)
        )
        for band, collection in enumerate(ax.collections):
            np.testing.assert_allclose(collection.get_offsets()[:, 1], evals[:, band])
            np.testing.assert_allclose(collection.get_array(), expected[:, band])
    else:
        for band, line in enumerate(ax.lines[: model.nstate]):
            np.testing.assert_allclose(line.get_ydata(), evals[:, band])


def test_orbital_projection_takes_precedence(ssh_model):
    # Spinless models still allow orbital projection when proj_spin is also set.
    fig, ax = ssh_model.plot_bands(
        K_NODES, nk=NK, proj_orb_idx=[0], proj_spin=True, cbar=False
    )

    assert len(fig.axes) == 1
    assert len(ax.collections) == ssh_model.nstate


@pytest.mark.parametrize("function", [plot_bands, TBModel.plot_bands])
def test_documented_parameters_match_signature(function):
    doc = inspect.getdoc(function)
    parameter_docs = doc.split("Parameters\n----------\n", 1)[1].split(
        "\nReturns\n-------\n", 1
    )[0]
    documented = re.findall(r"^(\w+) :", parameter_docs, flags=re.MULTILINE)
    parameters = [
        name
        for name in inspect.signature(function).parameters
        if name not in ("self", "model")
    ]

    assert len(documented) == len(parameters)
    assert set(documented) == set(parameters)
    if function is TBModel.plot_bands:
        assert documented == parameters
    assert "\nReturns\n-------\n" in doc


def test_model_copies_visualization_docstring():
    assert TBModel.plot_bands.__doc__ == plot_bands.__doc__
