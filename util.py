"""
util.py

Implementation of utility functions for ase.atoms.Atoms objects.
"""

from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.io.jsonio import decode
from ase.visualize import view
from ase.vibrations import Vibrations
from mace.calculators import MACECalculator
from tqdm import tqdm


def sample_to_atoms(sample, free_atoms_template, free_indices=None):
    """
    Convert a 1D sample vector to an Atoms object of the free subset.

    free_indices is optional: if not provided, it will be read from
    free_atoms_template.info['indices'] when available.
    """
    # infer free_indices if not provided
    if free_indices is None:
        if (
            free_atoms_template.info is None
            or "indices" not in free_atoms_template.info
        ):
            raise ValueError(
                "free_indices must be provided either as an argument or present in free_atoms_template.info['indices']"
            )
        free_indices = free_atoms_template.info["indices"]

    n_free = len(free_atoms_template)
    reshaped = np.asarray(sample, dtype=float).reshape(n_free, 3)
    out = free_atoms_template.copy()
    out.set_positions(reshaped)
    # store indices order
    info = dict(out.info) if out.info is not None else {}
    info["indices"] = list(free_indices)
    out.info = info
    return out


def combine_fixed_frozen_and_free_atoms(
    founder_atoms,
    fixed_atoms,
    fixed_indices,
    free_atoms,
    frozen_atoms=None,
    frozen_indices=None,
    calc=None,
):
    """
    Reconstruct a full 'Atoms' object from free-subset 'free_atoms' by
    inserting frozen and fixed atoms at the supplied indices.

    Args:
        founder_atoms: full Atoms used for cell, pbc and as fallback.
        fixed_atoms: Atoms containing the fixed atoms (order must match
            'fixed_indices'). If None, fixed positions are taken from founder.
        fixed_indices: indices in the full Atoms.
        free_atoms: Atoms containing only the free subset positions (in
            the ordering that will match insertion into the full array).
        frozen_atoms: (optional) Atoms containing frozen atoms (order must
            match 'frozen_indices').
        frozen_indices: (optional) indices for frozen atoms in the full Atoms.
        calc: optional ASE calculator to attach to the returned Atoms.

    Returns:
        combined_atoms: full Atoms with positions reconstructed.
    """

    if free_atoms is None:
        raise ValueError("free_atoms must be provided")

    if frozen_indices is None:
        frozen_indices = []
    if fixed_indices is None:
        fixed_indices = []

    if frozen_atoms is not None and len(frozen_indices) != len(frozen_atoms):
        raise ValueError(
            "frozen_indices length must equal frozen_atoms length."
        )

    if fixed_atoms is not None and len(fixed_indices) != len(fixed_atoms):
        raise ValueError("fixed_indices length must equal fixed_atoms length.")

    n_atoms = len(founder_atoms)
    combined_positions = np.zeros((n_atoms, 3), dtype=float)

    def _assign_positions(indices, source_atoms):
        if not indices:
            return
        src_pos = np.asarray(source_atoms.get_positions())
        if src_pos.shape[0] != len(indices):
            raise ValueError(
                f"Source positions length ({src_pos.shape[0]}) does not match indices length ({len(indices)})"
            )
        for i, idx in enumerate(indices):
            combined_positions[int(idx)] = src_pos[i]

    if fixed_indices:
        if fixed_atoms is not None:
            _assign_positions(fixed_indices, fixed_atoms)
        else:
            founder_pos = founder_atoms.get_positions()
            for idx in fixed_indices:
                combined_positions[int(idx)] = founder_pos[int(idx)]

    if frozen_indices:
        if frozen_atoms is None:
            # if frozen_indices exist but no frozen_atoms object provided, take from founder
            founder_pos = founder_atoms.get_positions()
            for idx in frozen_indices:
                combined_positions[int(idx)] = founder_pos[int(idx)]
        else:
            _assign_positions(frozen_indices, frozen_atoms)

    # Assign free atoms: infer free_indices order if needed
    if free_atoms is not None:
        if (
            hasattr(free_atoms, "info")
            and free_atoms.info is not None
            and "indices" in free_atoms.info
        ):
            free_indices = free_atoms.info["indices"]
        elif "free_indices" in locals() and free_indices is not None:
            free_indices = free_indices
        else:
            raise ValueError(
                "free_indices must be provided either as an argument or present in free_atoms.info['indices']"
            )

        free_pos = np.asarray(free_atoms.get_positions())
        if free_pos.shape[0] != len(free_indices):
            raise ValueError(
                f"free_atoms positions length ({free_pos.shape[0]}) does not match free_indices length ({len(free_indices)})"
            )
        for i, idx in enumerate(free_indices):
            combined_positions[int(idx)] = free_pos[i]

    founder_pos = founder_atoms.get_positions()
    for i in range(n_atoms):
        if np.allclose(combined_positions[i], 0.0):
            combined_positions[i] = founder_pos[i]

    combined_atoms = Atoms(
        symbols=founder_atoms.get_chemical_symbols(),
        positions=combined_positions,
        cell=founder_atoms.get_cell(),
        pbc=founder_atoms.get_pbc(),
    )

    if calc is not None:
        combined_atoms.calc = calc

    return combined_atoms


def solutions_to_atoms_list(
    solutions,
    founder_atoms,
    fixed_atoms,
    fixed_indices,
    free_atoms,
    calc,
    frozen_atoms=None,
    frozen_indices=None,
    free_indices=None,
):
    """
    Build full Atoms for each sample by overwriting free_indices (from samples)
    and frozen_indices (from relaxed frozen atoms of current gen).
    """
    if free_indices is None:
        if free_atoms.info is None or "indices" not in free_atoms.info:
            raise ValueError(
                "free_indices not provided and not in free_atoms.info"
            )
        free_indices = free_atoms.info["indices"]

    atoms_list = []
    for sample in solutions:
        free_subset = sample_to_atoms(sample, free_atoms, free_indices)
        combined = combine_fixed_frozen_and_free_atoms(
            founder_atoms=founder_atoms,
            fixed_atoms=fixed_atoms,
            fixed_indices=fixed_indices,
            free_atoms=free_subset,
            frozen_atoms=frozen_atoms,
            frozen_indices=frozen_indices,
            calc=calc,
        )
        atoms_list.append(combined)
    return atoms_list


def calc_atoms_list(
    atoms_list, func, desc, multiproc, n_proc, progress_bar, kwargs
):
    calc_atoms_list = []

    if progress_bar:
        if multiproc:
            with (
                Pool(processes=n_proc) as pool,
                tqdm(total=len(atoms_list), desc=desc) as pbar,
            ):
                for relaxed_atoms in pool.imap(
                    partial(func, **kwargs), atoms_list
                ):
                    calc_atoms_list.append(relaxed_atoms)
                    pbar.update()
                    pbar.refresh()
        else:
            for atoms in tqdm(atoms_list, desc=desc):
                calc_atoms_list.append(func(atoms, **kwargs))
    else:
        if multiproc:
            with Pool(processes=n_proc) as pool:
                for relaxed_atoms in pool.imap(
                    partial(func, **kwargs), atoms_list
                ):
                    calc_atoms_list.append(relaxed_atoms)
        else:
            for atoms in atoms_list:
                calc_atoms_list.append(func(atoms, **kwargs))

    return calc_atoms_list


def atoms_list_to_solutions(atoms_list, free_indices, dimensions=None):
    """
    Extract flattened positions for free indices across a list of Atoms.
    """
    free_atoms_list = [atoms[free_indices].copy() for atoms in atoms_list]
    positions = np.array([a.get_positions() for a in free_atoms_list])
    if dimensions is None:
        dimensions = positions.shape[1] * 3
    solutions = positions.reshape(-1, dimensions)
    return torch.from_numpy(solutions)


def init_calc(calc_str, device="cpu"):
    calc = None
    if calc_str == "LJ":
        calc = LennardJones(sigma=1 / (2 ** (1.0 / 6.0)), rc=10.0, smooth=True)
    else:
        calc = MACECalculator(
            model_paths=calc_str,
            device=device,
            default_dtype="float64",
        )

    return calc


def get_potential_energy(atoms, kwargs=None):
    return atoms.get_potential_energy()


def evaluate_population_with_calc(
    population: torch.Tensor,
    obj_params: dict,
    filter: bool = True,
    show: bool = False,
    bad_energy: float = -1e6,
) -> torch.Tensor:

    founder_atoms = decode(obj_params["founder_atoms"])
    fixed_atoms = decode(obj_params["fixed_atoms"])
    free_atoms = decode(obj_params["free_atoms"])

    frozen_atoms_raw = obj_params.get("frozen_atoms", None)
    frozen_atoms = (
        decode(frozen_atoms_raw) if frozen_atoms_raw is not None else None
    )
    frozen_indices = obj_params.get("frozen_indices", [])

    free_indices = obj_params.get("free_indices")
    if free_indices is None:
        if free_atoms.info is None or "indices" not in free_atoms.info:
            raise ValueError(
                "free_indices missing: set obj_params['free_indices'] or free_atoms.info['indices']."
            )
        free_indices = free_atoms.info["indices"]

    calc = init_calc(obj_params["calc"], obj_params["device"])

    atoms_list = obj_params.get("relaxed_atoms_list", None)
    if atoms_list is None:
        atoms_list = solutions_to_atoms_list(
            solutions=population,
            founder_atoms=founder_atoms,
            fixed_atoms=fixed_atoms,
            fixed_indices=obj_params["fixed_indices"],
            free_atoms=free_atoms,
            calc=calc,
            frozen_atoms=frozen_atoms,
            frozen_indices=frozen_indices,
        )

    energies = calc_atoms_list(
        atoms_list,
        func=get_potential_energy,
        desc="Evaluation",
        multiproc=obj_params["multiproc"],
        n_proc=obj_params["n_proc"],
        progress_bar=obj_params["progress_bar"],
        kwargs={},
    )

    if filter:
        updated_energies = []
        for atoms, energy in zip(atoms_list, energies):
            if energy < obj_params["e_cutoff"]:
                updated_energies.append(np.abs(bad_energy * energy))
            else:
                updated_energies.append(energy)
        energies = updated_energies

    if show:
        atoms_list_sorted = [
            atoms for _, atoms in sorted(zip(energies, atoms_list))
        ]
        view(atoms_list_sorted)

    return torch.Tensor(energies)


def check_saddle_point(atoms, eval_tol=0.1):
    """
    Quantifies saddle point quality using both mass-weighted frequencies 
    and raw Hessian eigenvalues.
    
    Safe for multiprocessing - uses temporary directories to avoid cache conflicts.
    """
    import tempfile
    import os
    from pathlib import Path
    
    # Create a temporary directory for this process's vibration cache
    with tempfile.TemporaryDirectory() as tmpdir:
        # Change to temp directory to avoid cache conflicts in multiprocessing
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            # 1. Run the vibration analysis
            vib = Vibrations(atoms)
            vib.run()
            
            # 2. Vibrational Frequency Analysis (Mass-Weighted)
            freqs = vib.get_frequencies()
            # In ASE, imaginary frequencies are returned as complex numbers (e.g., 0+500j)
            imaginary_freqs = [f for f in freqs if np.iscomplex(f)]
            
            # 3. Raw Hessian Analysis (Curvature only, no mass)
            hessian_raw = vib.get_vibrations().get_hessian_2d()

            # Eigenvalues of the raw Hessian (units: eV/Angstrom^2)
            raw_evals = np.linalg.eigvalsh(hessian_raw)
            
            # Count negative eigenvalues (ignoring tiny numerical noise near 0)
            # A threshold of -1e-5 is usually safe to separate noise from real curvature
            negative_evals = [val for val in raw_evals if val < -eval_tol]
            
            results = {
                "is_saddle_point": len(negative_evals) == 1,
                
                # Vibrational data
                "n_imaginary_freqs": len(imaginary_freqs),
                "imaginary_freq_val": imaginary_freqs[0] if imaginary_freqs else None,
                "all_freqs_cm1": np.round(freqs, 3),
                
                # Raw Hessian data
                "n_negative_evals": len(negative_evals),
                "raw_eigenvalues": np.round(raw_evals, 3),
                "lowest_eval": raw_evals[0]
            }
            
            # Clean up ASE temporary files
            vib.clean()
            
            return results
        
        finally:
            # Always return to original directory
            os.chdir(original_cwd)


def get_lowest_eigenvalue(atoms, kwargs=None):
    """
    Helper function to extract lowest eigenvalue from an atoms object.
    Used for multiprocessing in minimize_eigenval.
    """
    return check_saddle_point(atoms)["lowest_eval"]


def minimize_eigenval(
    population: torch.Tensor,
    obj_params: dict,
    filter: bool = True,
    show: bool = False,
    bad_eigenval: float = 1e6,
) -> torch.Tensor:
    """
    Evaluate population based on minimizing the lowest eigenvalue from saddle point analysis.
    Works like evaluate_population_with_calc but uses the 'lowest_eval' result from 
    check_saddle_point function.
    
    This objective function is useful for evolutionary algorithms targeting transition state 
    optimization, where you want to maximize the negative curvature (minimize eigenvalue).
    
    Args:
        population: torch.Tensor of candidate solutions
        obj_params: dict with structure parameters (founder_atoms, fixed_atoms, free_atoms, etc.)
        filter: if True, penalize structures with eigenvalue > eval_cutoff
        show: if True, visualize the structures sorted by eigenvalue
        bad_eigenval: penalty multiplier for filtered structures (default 1e6)
    
    Returns:
        torch.Tensor of eigenvalue objectives (to be minimized)
    """
    founder_atoms = decode(obj_params["founder_atoms"])
    fixed_atoms = decode(obj_params["fixed_atoms"])
    free_atoms = decode(obj_params["free_atoms"])

    frozen_atoms_raw = obj_params.get("frozen_atoms", None)
    frozen_atoms = (
        decode(frozen_atoms_raw) if frozen_atoms_raw is not None else None
    )
    frozen_indices = obj_params.get("frozen_indices", [])

    free_indices = obj_params.get("free_indices")
    if free_indices is None:
        if free_atoms.info is None or "indices" not in free_atoms.info:
            raise ValueError(
                "free_indices missing: set obj_params['free_indices'] or free_atoms.info['indices']."
            )
        free_indices = free_atoms.info["indices"]

    calc = init_calc(obj_params["calc"], obj_params["device"])

    atoms_list = obj_params.get("relaxed_atoms_list", None)
    if atoms_list is None:
        atoms_list = solutions_to_atoms_list(
            solutions=population,
            founder_atoms=founder_atoms,
            fixed_atoms=fixed_atoms,
            fixed_indices=obj_params["fixed_indices"],
            free_atoms=free_atoms,
            calc=calc,
            frozen_atoms=frozen_atoms,
            frozen_indices=frozen_indices,
        )

    # Evaluate lowest eigenvalue for each structure
    eigenvalues = calc_atoms_list(
        atoms_list,
        func=get_lowest_eigenvalue,
        desc="Eigenvalue Evaluation",
        multiproc=obj_params["multiproc"],
        n_proc=obj_params["n_proc"],
        progress_bar=obj_params["progress_bar"],
        kwargs={},
    )

    if filter:
        eval_cutoff = obj_params.get("eval_cutoff", -0.1)
        updated_eigenvalues = []
        for atoms, eigenval in zip(atoms_list, eigenvalues):
            if eigenval < eval_cutoff:
                # Good saddle point: minimize the eigenvalue
                updated_eigenvalues.append(eigenval)
            else:
                # Poor saddle point: penalize with large positive value
                updated_eigenvalues.append(bad_eigenval)
        eigenvalues = updated_eigenvalues

    if show:
        atoms_list_sorted = [
            atoms for _, atoms in sorted(zip(eigenvalues, atoms_list))
        ]
        view(atoms_list_sorted)

    return torch.Tensor(eigenvalues)


