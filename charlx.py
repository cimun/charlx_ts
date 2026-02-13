"""
charlx.py

Implementation of the AtomsFixer, AtomsRelaxer and CHARLX classes.
"""

import os
from typing import Optional

import numpy as np
import torch
from ase.constraints import FixAtoms
from ase.io.jsonio import encode
from ase.optimize import BFGS, FIRE
from condevo.es import CHARLES
from condevo.es.traditional_es import CMAES

from cluster import get_distances_from_scaled_position
from util import (
    atoms_list_to_solutions,
    calc_atoms_list,
    init_calc,
    solutions_to_atoms_list,
)


class AtomsFixer:
    """
    AtomsFixer class with support for:
    - fixed atoms (always constrained, never move)
    - frozen atoms (not modified during candidate generation, but can relax)
    - free atoms (modifiable and relaxable)
    """

    def __init__(
        self,
        fix_func=None,
        fix_list=None,
        freeze_func=None,
        freeze_list=None,
        fix_axis=None,
        fix_threshold=None,
        freeze_axis=None,
        freeze_threshold=None,
        fix_center=None,
        fix_radius=None,
    ):
        # Fixed atoms
        self.fix_func = fix_func
        self.fix_list = fix_list or []
        self.fix_axis = fix_axis
        self.fix_threshold = fix_threshold
        self.fix_center = fix_center
        self.fix_radius = fix_radius

        # Frozen atoms
        self.freeze_func = freeze_func
        self.freeze_list = freeze_list or []
        self.freeze_axis = freeze_axis
        self.freeze_threshold = freeze_threshold

        # Fixed dispatchers
        if fix_func == "get_fixed_by_list":
            self.get_fixed_indices = self.get_fixed_by_list
        elif fix_func == "get_fixed_by_threshold":
            self.get_fixed_indices = self.get_fixed_by_threshold
        elif fix_func == "get_fixed_by_radius":
            self.get_fixed_indices = self.get_fixed_by_radius
        else:
            self.get_fixed_indices = lambda atoms: []

        # Frozen dispatchers
        if freeze_func == "get_frozen_by_list":
            self.get_frozen_indices = self.get_frozen_by_list
        elif freeze_func == "get_frozen_by_threshold":
            self.get_frozen_indices = self.get_frozen_by_threshold
        else:
            self.get_frozen_indices = lambda atoms: []

    # Fixed atoms
    def get_fixed_by_list(self, atoms):
        return [atom.index for atom in atoms if atom.index in self.fix_list]

    def get_fixed_by_threshold(self, atoms):
        return [
            atom.index
            for atom in atoms
            if atom.position[self.fix_axis] < self.fix_threshold
        ]

    def get_fixed_by_radius(self, atoms):
        dist = get_distances_from_scaled_position(
            atoms=atoms, scaled_position=self.fix_center
        )
        fixed_indices = np.where(dist > self.fix_radius)[0].tolist()
        return fixed_indices

    # Frozen atoms
    def get_frozen_by_list(self, atoms):
        return [atom.index for atom in atoms if atom.index in self.freeze_list]

    def get_frozen_by_threshold(self, atoms):
        return [
            atom.index
            for atom in atoms
            if atom.position[self.freeze_axis] < self.freeze_threshold
        ]

    # Combined
    def get_indices(self, atoms):
        fixed_indices = list(self.get_fixed_indices(atoms))
        frozen_indices = list(self.get_frozen_indices(atoms))
        fixed_set = set(fixed_indices)
        frozen_set = set(frozen_indices) - fixed_set
        all_constrained = fixed_set | frozen_set
        free_indices = [
            atom.index for atom in atoms if atom.index not in all_constrained
        ]
        fixed_indices = sorted(fixed_set)
        frozen_indices = sorted(frozen_set)
        free_indices = sorted(free_indices)
        return fixed_indices, frozen_indices, free_indices

    def __str__(self):
        return (
            f"AtomsFixer(fix_func={self.fix_func}, fix_list={self.fix_list}, "
            f"fix_axis={self.fix_axis}, fix_threshold={self.fix_threshold}, "
            f"freeze_func={self.freeze_func}, freeze_list={self.freeze_list}, "
            f"freeze_axis={self.freeze_axis}, freeze_threshold={self.freeze_threshold}, fix_center={self.fix_center}, fix_radius={self.fix_radius})"
        )

    __repr__ = __str__


class AtomsRelaxer:
    """
    Applies only fixed constraints during relaxation.
    Frozen atoms are excluded from sampling but can move in relaxation.
    """

    def __init__(
        self,
        founder_atoms,
        fixer: AtomsFixer,
        calc: str,
        optimizer: str,
        fmax: float,
        steps: int,
        logfile: str,
        multiproc: bool,
        n_proc: int,
        device: str,
        e_cutoff: float,
        progress_bar: bool = False,
        save_traj: bool = False,
        traj_path: Optional[str] = None,
        save_interval: Optional[int] = 10,
    ):
        self.founder_atoms = founder_atoms
        self.n_atoms = len(founder_atoms)

        # Get indices
        self.fixed_indices, self.frozen_indices, self.free_indices = (
            fixer.get_indices(founder_atoms)
        )

        # Subsets
        self.fixed_atoms = (
            founder_atoms[self.fixed_indices].copy()
            if self.fixed_indices
            else None
        )
        self.frozen_atoms = (
            founder_atoms[self.frozen_indices].copy()
            if self.frozen_indices
            else None
        )
        self.free_atoms = founder_atoms[self.free_indices].copy()

        # Attach indices info to free_atoms
        info = (
            dict(self.free_atoms.info)
            if self.free_atoms.info is not None
            else {}
        )
        info["indices"] = list(self.free_indices)
        self.free_atoms.info = info

        # ES dimension = free only
        self.free_n_atoms = len(self.free_atoms)
        self.free_positions = torch.from_numpy(
            self.free_atoms.get_positions().flatten()
        )
        self.dimensions = self.free_positions.shape[0]

        # Movable in relaxation = free + frozen
        self.movable_indices = self.free_indices + self.frozen_indices
        self.movable_atoms = founder_atoms[self.movable_indices].copy()
        self.movable_dim = len(self.movable_indices) * 3

        # Relaxation params
        self.calc_str = calc
        self.calc = init_calc(calc, device)
        self.fmax = fmax
        self.steps = steps
        self.logfile = logfile
        self.multiproc = multiproc
        self.n_proc = n_proc
        self.device = device
        self.e_cutoff = e_cutoff
        self.progress_bar = progress_bar

        if optimizer == "FIRE":
            self.optimizer = FIRE
        elif optimizer == "BFGS":
            self.optimizer = BFGS
        else:
            self.optimizer = None

        self.save_traj = save_traj
        self.traj_path = traj_path
        if self.save_traj and self.traj_path is not None:
            os.makedirs(self.traj_path, exist_ok=True)
        self.save_interval = save_interval

    def refresh_indices(self, solutions):
        atoms_list = solutions_to_atoms_list(
            solutions=solutions,
            founder_atoms=self.founder_atoms,
            fixed_atoms=self.fixed_atoms,
            fixed_indices=self.fixed_indices,
            frozen_atoms=self.frozen_atoms,
            frozen_indices=self.frozen_indices,
            free_atoms=self.free_atoms,
            calc=self.calc,
        )
        if self.frozen_indices or self.fixed_indices:
            for atoms in atoms_list:
                all_constrained = list(
                    set(self.fixed_indices + self.frozen_indices)
                )
                atoms.set_constraint(FixAtoms(indices=all_constrained))
        solutions = atoms_list_to_solutions(
            atoms_list=atoms_list,
            free_indices=self.free_indices,
            dimensions=self.dimensions,
        )
        return solutions

    def relax(self, solutions, gen=None):
        atoms_list = solutions_to_atoms_list(
            solutions=solutions,
            founder_atoms=self.founder_atoms,
            fixed_atoms=self.fixed_atoms,
            fixed_indices=self.fixed_indices,
            frozen_atoms=self.frozen_atoms,
            frozen_indices=self.frozen_indices,
            free_atoms=self.free_atoms,
            calc=self.calc,
            free_indices=self.free_indices,
        )
        relaxed_atoms_list = calc_atoms_list(
            atoms_list=atoms_list,
            func=self.relax_atoms,
            desc="Relaxation",
            multiproc=self.multiproc,
            n_proc=self.n_proc,
            progress_bar=self.progress_bar,
            kwargs={"steps": self.steps, "gen": gen},
        )
        solutions = atoms_list_to_solutions(
            atoms_list=relaxed_atoms_list,
            free_indices=self.free_indices,
            dimensions=self.dimensions,
        )
        return solutions, relaxed_atoms_list

    def relax_atoms(
        self,
        atoms,
        steps,
        gen: Optional[int] = None,
        idx: int = 0,
        save_interval: Optional[int] = 10,
    ):
        if self.fixed_indices:
            atoms.set_constraint(FixAtoms(indices=self.fixed_indices))
        atoms.wrap()

        # If saving trajectories
        if save_interval < steps:
            save_interval = steps

        if self.save_traj and self.traj_path is not None and gen is not None:
            os.makedirs(self.traj_path, exist_ok=True)
            traj_filename = os.path.join(self.traj_path, f"gen_{gen}.traj")
            dyn = self.optimizer(
                atoms=atoms,
                logfile=self.logfile,
                trajectory=traj_filename,
                append_trajectory=True,
            )
        else:
            dyn = self.optimizer(atoms=atoms, logfile=self.logfile)

        dyn.run(fmax=self.fmax, steps=steps)
        dyn.atoms.wrap()
        return dyn.atoms

    def encode_params(self):
        params = {
            "founder_atoms": encode(self.founder_atoms),
            "fixed_atoms": encode(self.fixed_atoms),
            "frozen_atoms": encode(self.frozen_atoms),
            "free_atoms": encode(self.free_atoms),
            "fixed_indices": self.fixed_indices,
            "frozen_indices": self.frozen_indices,
            "free_indices": self.free_indices,
            "calc": self.calc_str,
            "multiproc": self.multiproc,
            "n_proc": self.n_proc,
            "progress_bar": self.progress_bar,
            "device": self.device,
            "e_cutoff": self.e_cutoff,
        }
        return params

    def __str__(self):
        return f"AtomsRelaxer(founder_atoms={self.founder_atoms}, fixed_indices={self.fixed_indices}, frozen_indices={self.frozen_indices}, free_indices={self.free_indices}, calc={self.calc}, optimizer={self.optimizer}, fmax={self.fmax:.5f}, steps={self.steps}, logfile={self.logfile}, multiproc={self.multiproc}, n_proc={self.n_proc}, device={self.device}, save_traj={self.save_traj}, traj_path={self.traj_path}, save_interval={self.save_interval})"

    __repr__ = __str__


class CHARLX(CHARLES):
    """
    CHARLX with support for fixed (always constrained) and frozen (sampling-only) atoms.
    """

    def __init__(self, fixer, relaxer, n_gens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixer = fixer
        self.relaxer = relaxer
        self.relaxed_atoms_list = None
        self.n_gens = n_gens
        self.curr_gen = 0

    def ask(self):
        self.solutions = super().ask()
        self.solutions = self.relaxer.refresh_indices(self.solutions)
        self.solutions, self.relaxed_atoms_list = self.relaxer.relax(
            self.solutions, gen=self.curr_gen
        )
        self.curr_gen += 1
        return self.solutions
