"""
cluster.py

Implementation of Lennard-Jones cluster functionalities.

Code adapted from Clinamen2 [1].

[1] https://github.com/Madsen-s-research-group/clinamen2-public-releases
"""

import pathlib
from typing import Optional, Tuple

import ase
import numpy as np
import numpy.typing as npt
import torch


def get_distances_from_position(
    atoms: ase.atoms.Atoms, position: Tuple[float, float, float]
):
    """Get distances to position in Angstrom.

    Args:
        atoms: Atoms object to calculate distances for.
        position: Position to calculate distance from.
    """

    atoms_copy = atoms.copy()
    atom_at_position = ase.atoms.Atom(symbol="H", position=position)
    atoms_copy.append(atom_at_position)

    # get distance to atom at given position for all atoms
    distances = atoms_copy.get_all_distances(mic=True)[-1]

    return distances[:-1]  # exclude defect atom


def get_distances_from_scaled_position(
    atoms: ase.atoms.Atoms, scaled_position: Tuple[float, float, float]
):
    """Wrapper for get_distances_from_position().

    Args:
        atoms: Atoms object to calculate distances for.
        scaled_position: Position to calculate distance from.
    """
    return get_distances_from_position(
        atoms=atoms, position=scaled_position @ atoms.cell
    )


def create_lj_cluster(
    n_atoms: int = 38,
    identifier: Optional[str] = None,
    wales_path: pathlib.Path = None,
) -> npt.ArrayLike:
    """
    Create an LJ cluster from a Cambridge database entry

    Load the coordinates of the ground state of an n- atom LJ cluster
    from an entry in

    http://doye.chem.ox.ac.uk/jon/structures/LJ.html

    and return the corresponding LennardJones object.

    Args:
        n_atoms: Number of atoms in the cluster.
        identifier: Additional identifier of a specific configuration.
            For example "i" for "38i". Default is None.
        wales_path: Path to the Wales potential data.

    Returns:
        npt.ArrayLike
            - Coordinates of specified Cluster

    Raises:
        ValueError: If there is a problem with the argument.

    References:

        [1] The Cambridge Cluster Database, D. J. Wales, J. P. K. Doye,
        A. Dullweber, M. P. Hodges, F. Y. Naumkin F. Calvo, J. Hernández-Rojas
        and T. F. Middleton, URL http://www-wales.ch.cam.ac.uk/CCD.html.
    """

    if isinstance(n_atoms, int) or isinstance(n_atoms, str):
        cluster = (
            str(n_atoms) if identifier is None else str(n_atoms) + identifier
        )
        filename = wales_path / cluster
    else:
        raise ValueError("n must be an integer or a string")

    if not filename.exists():
        raise ValueError(
            f"Coordinates for {n_atoms} atoms with identifier"
            f" {identifier} not found in {wales_path}."
        )

    coordinates = np.loadtxt(filename)
    # These coordinates use the sigma = 1 convention -> rescale
    coordinates /= 2 ** (1.0 / 6.0)

    return coordinates


def get_max_span_from_lj_cluster(
    lj_cluster: npt.ArrayLike,
    verbose=False,
) -> float:
    """
    Evaluate LJ cluster positions and return largest span.

    Args:
        lj_cluster: Positions of the LJ spheres.
    """

    lj_cluster_resh = np.reshape(lj_cluster, newshape=(-1, 3))
    max_x_span = abs(lj_cluster_resh[:, 0].min() - lj_cluster_resh[:, 0]).max()
    max_y_span = abs(lj_cluster_resh[:, 1].min() - lj_cluster_resh[:, 1]).max()
    max_z_span = abs(lj_cluster_resh[:, 2].min() - lj_cluster_resh[:, 2]).max()

    if verbose:
        print(f"max x span = {max_x_span}")
        print(f"max y span = {max_y_span}")
        print(f"max z span = {max_z_span}")

    return np.max([max_x_span, max_y_span, max_z_span])


def place_atoms_random_sphere(
    n_atoms: int = 38,
    radius: float = 3.5,
    random_seed: int = 0,
):
    """
    Place atoms randomly within a sphere of a given radius.

    The sphere is always centered at zero.

    Args:
        n_atoms: Number of atoms to be placed.
        radius: Radius of the sphere.
        random_seed: Seed for the random number generator. Default is 0.
    """

    rng = np.random.default_rng(seed=random_seed)
    phi = rng.uniform(low=0.0, high=2.0 * np.pi, size=n_atoms)
    costheta = rng.uniform(low=-1.0, high=1.0, size=n_atoms)
    u = rng.random(size=n_atoms)
    theta = np.arccos(costheta)
    r = radius * np.cbrt(u)

    points = np.asarray(
        [
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]
    ).T.flatten()

    return points


def load_lj_cluster(
    n_atoms: int,
    identifier: Optional[str] = None,
    wales_path: pathlib.Path = None,
    randomize: bool = False,
    random_seed: int = 0,
    element: str = "Ar",
    name: str = "LJ",
) -> Tuple[npt.ArrayLike, float, int]:
    """
    load_lj_cluster() function docstring
    """

    lj_cluster = create_lj_cluster(
        n_atoms=n_atoms,
        identifier=identifier,
        wales_path=wales_path,
    )

    max_span = get_max_span_from_lj_cluster(
        lj_cluster=lj_cluster,
        verbose=False,
    )

    dimensions = n_atoms * 3

    if randomize:
        lj_cluster = place_atoms_random_sphere(
            n_atoms=n_atoms,
            radius=max_span,
            random_seed=random_seed,
        )

    symbols = f"{element}{n_atoms}"
    rep = f"{name}-{n_atoms}"

    return lj_cluster, max_span, dimensions, symbols, rep


def evaluate_lj_population_torch(
    population: torch.Tensor,
    obj_params: dict = None,
) -> torch.Tensor:
    """
    Calculate LJ energy of a population of clusters with torch.

    Args:
        population: torch.Tensor containing the population.

    Returns:
        energies: torch.Tensor with the energies of the population.
    """

    # Calculate number of atoms for every cluster
    n_atoms = int(population.size(1) / 3)

    # Reshape tensor to (x,y,z) coordinates
    positions = population.reshape((-1, n_atoms, 3))

    # Compute all distances between pairs without iterating
    delta = positions[:, :, None, :] - positions[:, None]
    r2 = (delta * delta).sum(axis=3)

    # Take only the upper triangle (combinations of two atoms)
    rows, cols = torch.triu_indices(row=r2.size(1), col=r2.size(2), offset=1)
    rm2 = 1.0 / r2[:, rows, cols]

    # Compute the potental energy recycling some calculations
    rm6 = rm2 * rm2 * rm2
    energies = (rm6 * (rm6 - 2.0)).sum(axis=1)

    return energies
