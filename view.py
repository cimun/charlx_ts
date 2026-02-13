"""
view.py

Implementation of visualization and post-processing functionalities.
"""

import json
import os
from pprint import pprint

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ase import Atoms
from ase.io import read, write
from ase.io.jsonio import decode
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from matplotlib.ticker import MaxNLocator

from run import H5_FILE
from util import (
    combine_fixed_frozen_and_free_atoms,
    init_calc,
    sample_to_atoms,
)


def load_benchmark(
    es="HADES", objective="rastrigin", run_ids=None, dst="output/"
):
    """load run data as pandas dataframe"""
    h5_filename = os.path.join(dst, H5_FILE.format(ES=es, objective=objective))

    h5_files = []
    h5_basename = os.path.basename(h5_filename.replace(".h5", ""))
    for filename in os.listdir(os.path.dirname(h5_filename)):
        if h5_basename in filename:
            h5_files.append(
                os.path.join(os.path.dirname(h5_filename), filename)
            )

    h5_data = []
    run_offset = 0
    for h5_filename in h5_files:
        print("loading data from", h5_filename)
        with h5py.File(h5_filename, "r") as f:
            if run_ids is None:
                run_ids = list(
                    sorted([int(k.replace("run_", "")) for k in f.keys()])
                )

            if not isinstance(run_ids, list):
                run_ids = [run_ids]

            data = []
            for run_id in run_ids:
                run_data = f[f"run_{run_id}"]
                gen_ids = list(
                    sorted(
                        [int(k.replace("gen_", "")) for k in run_data.keys()]
                    )
                )

                # extract attributes
                run_attrs = run_data.attrs
                run_dict = {}
                for k, v in run_attrs.items():
                    try:
                        v = json.loads(v)
                    except:
                        pass
                    run_dict[k] = v

                for gen_id in gen_ids:
                    generation = run_data[f"gen_{gen_id}"]

                    try:
                        samples = generation["samples"][()]
                        fitness = generation["fitness"][()]

                        item = {
                            "run": run_id + run_offset,
                            "gen": gen_id,
                            "samples": samples,
                            "fitness": fitness,
                            **run_dict,
                        }
                        if "model_loss" in generation:
                            model_loss = generation["model_loss"][()]
                            item["model_loss"] = model_loss

                        # add generation data and attribute data as data-element for dataframe
                        data.append(item)

                    except KeyError:
                        pass

            run_offset += len(run_ids)
            h5_data.extend(data)

    return pd.DataFrame(h5_data)


def view_best_samples(
    df: pd.DataFrame,
    calc_str: str = None,
    calc_energy: bool = False,
    sort_samples: bool = False,
    show: bool = True,
    es_samples: bool = False,
    rlx_structures: bool = True,
):
    obj_params = df["objective"].iloc[0]["foo_kwargs"]["obj_params"]
    founder_atoms = decode(obj_params["founder_atoms"])

    if calc_energy:
        calc = init_calc(calc_str)

    list_of_best_atoms = []
    for gen, row in df.iterrows():
        fitness = row["fitness"]
        idx_best = np.argmax(fitness)
        energy = -fitness[idx_best]

        if es_samples:
            sample = row["samples"][idx_best]
            free_subset = sample_to_atoms(
                sample,
                decode(obj_params["free_atoms"]),
                obj_params["free_indices"],
            )
            atoms = combine_fixed_frozen_and_free_atoms(
                founder_atoms=founder_atoms,
                fixed_atoms=None,
                fixed_indices=obj_params.get("fixed_indices", []),
                free_atoms=free_subset,
                frozen_atoms=None,
                frozen_indices=obj_params.get("frozen_indices", []),
                calc=None,
            )
        elif rlx_structures and "relaxed_positions" in row:
            pos = row["relaxed_positions"][idx_best]
            atoms = founder_atoms.copy()
            atoms.set_positions(pos)

        list_of_best_atoms.append(atoms)

        if calc_energy:
            atoms.calc = calc
            mace_energy = atoms.get_potential_energy()
            print(
                f"Generation {gen+1}. Energy = {energy:.6f} eV, "
                f"MACE = {mace_energy:.6f} eV, "
                f"Diff = {np.abs(energy - mace_energy):.6f} eV"
            )
        else:
            print(f"Generation {gen+1}. Energy = {energy:.6f} eV")

    if show:
        view(list_of_best_atoms)


def view_generation_samples(
    df: pd.DataFrame,
    generation: int = -1,
    calc_str: str = None,
    calc_energy: bool = False,
    sort_samples: bool = False,
    show: bool = True,
    es_samples: bool = False,
    rlx_structures: bool = True,
):
    obj_params = df["objective"].iloc[0]["foo_kwargs"]["obj_params"]
    founder_atoms = decode(obj_params["founder_atoms"])
    if calc_energy:
        calc = init_calc(calc_str)

    gen_idx = generation if generation >= 0 else (len(df) - 1)
    row = df.iloc[gen_idx]
    fitness = row["fitness"]
    samples = row["samples"]
    relaxed_positions = row.get("relaxed_positions", None)

    if sort_samples:
        idx_order = np.argsort(fitness)[::-1]
        fitness = fitness[idx_order]
        samples = samples[idx_order]
        if relaxed_positions is not None:
            relaxed_positions = relaxed_positions[idx_order]

    print_gen = "last" if generation == -1 else generation + 1
    print(f"Generation {print_gen}:")

    list_of_atoms = []
    for index, fit in enumerate(fitness):
        energy = -fit
        if es_samples:
            sample = samples[index]
            free_subset = sample_to_atoms(
                sample,
                decode(obj_params["free_atoms"]),
                obj_params["free_indices"],
            )
            atoms = combine_fixed_frozen_and_free_atoms(
                founder_atoms=founder_atoms,
                fixed_atoms=None,
                fixed_indices=obj_params.get("fixed_indices", []),
                free_atoms=free_subset,
                frozen_atoms=None,
                frozen_indices=obj_params.get("frozen_indices", []),
                calc=None,
            )
        elif rlx_structures and relaxed_positions is not None:
            pos = relaxed_positions[index]
            atoms = founder_atoms.copy()
            atoms.set_positions(pos)

        list_of_atoms.append(atoms)
        if calc_energy:
            atoms.calc = calc
            mace_energy = atoms.get_potential_energy()
            print(
                f"Individuum {index+1}. Energy = {energy:.6f} eV, "
                f"MACE = {mace_energy:.6f} eV, "
                f"Diff = {np.abs(energy - mace_energy):.6f} eV"
            )
        else:
            print(f"Individuum {index+1}. Energy = {energy:.6f} eV")

    if show:
        view(list_of_atoms)


def attach_relaxed_positions_from_h5(df, dst, es, objective):
    """
    Load relaxed positions from H5 files and attach them to the dataframe.
    """
    try:
        h5_template = os.path.join(
            dst, H5_FILE.format(ES=es, objective=objective)
        )
        h5_dir = os.path.dirname(h5_template)
        h5_basename = os.path.basename(h5_template).replace(".h5", "")

        # List all files matching the pattern
        h5_files = [
            os.path.join(h5_dir, fn)
            for fn in os.listdir(h5_dir)
            if h5_basename in fn
        ]
    except Exception:
        return df

    # Map (global_run, gen) -> relaxed_positions array
    relaxed_map = {}
    run_offset = 0

    for h5_file in sorted(h5_files):
        try:
            with h5py.File(h5_file, "r") as f:
                runs = sorted([int(k.replace("run_", "")) for k in f.keys()])
                for run_id in runs:
                    run_group = f[f"run_{run_id}"]
                    gens = sorted(
                        [int(k.replace("gen_", "")) for k in run_group.keys()]
                    )

                    for gen_id in gens:
                        gen_group = run_group[f"gen_{gen_id}"]
                        if "relaxed_positions" in gen_group:
                            try:
                                rp = gen_group["relaxed_positions"][()]
                                relaxed_map[(run_id + run_offset, gen_id)] = rp
                            except Exception:
                                pass
                run_offset += len(runs)
        except Exception:
            continue

    if not relaxed_map:
        return df

    # Attach relaxed_positions to DataFrame rows where available
    if "relaxed_positions" not in df.columns:
        df["relaxed_positions"] = [None] * len(df)

    for idx, row in df.iterrows():
        key = (int(row["run"]), int(row["gen"]))
        if key in relaxed_map:
            df.at[idx, "relaxed_positions"] = relaxed_map[key]

    return df


def safe_decode(x):
    """
    Decodes an object if it is a string, otherwise returns it as is.
    """
    if x is None:
        return None
    try:
        return decode(x) if isinstance(x, str) else x
    except Exception:
        return x


def get_atom_radii(atoms, r=0.9):
    """
    Returns a list of radii for the atoms object for plotting.
    """
    if atoms is None:
        return None
    try:
        return [r] * len(atoms)
    except Exception:
        return None


def display_dataframe(df_summary):
    """
    Helper to print dataframe.
    """
    try:
        from IPython.display import display

        display(df_summary)
    except ImportError:
        print(df_summary.to_string())


def print_config_section(title, config_dict):
    print(f"\n=== {title} ===")
    for k, v in config_dict.items():
        print(f"{k:25}: {v}")


def view_results(
    dst,
    obj="evaluate_population_with_calc",
    es="CHARLX",
    gen=-1,
    run_ids=0,
    calc_energy=False,
    folder=None,
    structure_file=None,
    sort_samples: bool = False,
    show: bool = True,
    es_samples: bool = False,
    rlx_structures: bool = True,
):
    df = load_benchmark(
        objective=obj,
        es=es,
        dst=dst,
        run_ids=run_ids,
    )

    founder = decode(
        df["objective"].iloc[0]["foo_kwargs"]["obj_params"]["founder_atoms"]
    )
    calc = df["objective"].iloc[0]["foo_kwargs"]["obj_params"]["calc"]

    initial_atoms, initial_energy = None, None
    if folder and structure_file:
        structure_path = f"{folder}/{structure_file}"
        initial_atoms = read(structure_path)
        initial_atoms.calc = init_calc(calc)
        initial_energy = initial_atoms.get_potential_energy()
        print(f"Initial Structure Energy: {initial_energy:.4f} eV\n")
        view(initial_atoms)
    else:
        initial_atoms = founder
        initial_atoms.calc = init_calc(calc)
        initial_energy = initial_atoms.get_potential_energy()
        print(f"Initial Structure Energy: {initial_energy:.4f} eV\n")

    if "relaxed_positions" not in df.columns:
        df = attach_relaxed_positions_from_h5(df, dst, es, obj)

    es_config = df["es_config"].iloc[0]
    obj_params = df["objective"].iloc[0]["foo_kwargs"]["obj_params"]

    print_config_section("Evolution Strategy Config", es_config)
    print_config_section("Objective Parameters", obj_params)

    view_best_samples(
        df=df,
        calc_str=calc,
        calc_energy=calc_energy,
        sort_samples=sort_samples,
        es_samples=es_samples,
        rlx_structures=rlx_structures,
        show=show,
    )
    view_generation_samples(
        df=df,
        generation=gen,
        calc_str=calc,
        calc_energy=calc_energy,
        sort_samples=sort_samples,
        es_samples=es_samples,
        rlx_structures=rlx_structures,
        show=show,
    )

    if not isinstance(df, pd.DataFrame):
        print("Benchmark is not a DataFrame; skipping plots.")
        return

    # Calculate Best Energy (fitness is maximized; energy = -fitness)
    df["best_energy"] = df["fitness"].apply(lambda f: -max(f))

    # Plot - Energy Evolution
    plt.figure(figsize=(8, 4))
    sns.lineplot(
        x=np.arange(len(df)), y=df["best_energy"], marker="o", ci=None
    )
    plt.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.xlabel("Generation")
    plt.ylabel("Best Energy (eV)")
    plt.title("Best energy per generation")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # Table - Metrics Summary
    best_idx = df["best_energy"].idxmin()
    best_energy = df.loc[best_idx, "best_energy"]
    last_gen_idx = len(df) - 1 if gen == -1 else gen

    # Calculate delta E
    delta_e = (
        (initial_energy - df.loc[last_gen_idx, "best_energy"])
        if initial_energy is not None
        else None
    )

    summary = {
        "Initial energy (eV)": initial_energy,
        "Final best energy (eV)": df.loc[last_gen_idx, "best_energy"],
        "Improvement vs initial (ΔE, eV)": delta_e,
        "Global best energy (eV)": best_energy,
        "Best generation (1-based)": int(best_idx + 1),
        "Generations run": int(len(df)),
    }

    display_dataframe(pd.DataFrame(summary, index=["Results"]).T)

    try:
        # Decode parameters
        founder_atoms = safe_decode(obj_params.get("founder_atoms"))
        fixed_atoms = safe_decode(obj_params.get("fixed_atoms"))
        free_atoms = safe_decode(obj_params.get("free_atoms"))
        frozen_atoms = safe_decode(obj_params.get("frozen_atoms", None))

        fixed_indices = obj_params.get("fixed_indices", None)
        frozen_indices = obj_params.get("frozen_indices", None)
        free_indices = obj_params.get("free_indices", None)

        # # Debug prints
        # print("founder_atoms:", founder_atoms)
        # print("fixed_atoms:", fixed_atoms)
        # print("free_atoms:", free_atoms)
        # print("frozen_atoms:", frozen_atoms)

        # Reconstruct Best of Final Generation
        final_row = df.iloc[last_gen_idx]
        fitness = final_row["fitness"]
        samples = final_row["samples"]
        relaxed_positions = final_row.get("relaxed_positions", None)

        if sort_samples:
            idx_order = np.argsort(fitness)[::-1]
            fitness = fitness[idx_order]
            samples = samples[idx_order]
            if relaxed_positions is not None:
                relaxed_positions = relaxed_positions[idx_order]

        sample_idx = 0

        if es_samples and samples is not None:
            final_best_sample = samples[sample_idx]
        elif relaxed_positions is not None:
            final_best_sample = None
        else:
            final_best_sample = samples[sample_idx]

        if es_samples:
            final_best_atoms = combine_fixed_frozen_and_free_atoms(
                founder_atoms=founder_atoms,
                fixed_atoms=fixed_atoms,
                fixed_indices=fixed_indices,
                free_atoms=sample_to_atoms(final_best_sample, free_atoms),
                frozen_atoms=frozen_atoms,
                frozen_indices=frozen_indices,
                calc=None,
            )
        elif rlx_structures and relaxed_positions is not None:
            final_best_atoms = founder_atoms.copy()
            final_best_atoms.set_positions(relaxed_positions[sample_idx])
        else:
            final_best_atoms = combine_fixed_frozen_and_free_atoms(
                founder_atoms=founder_atoms,
                fixed_atoms=fixed_atoms,
                fixed_indices=fixed_indices,
                free_atoms=sample_to_atoms(final_best_sample, free_atoms),
                frozen_atoms=frozen_atoms,
                frozen_indices=frozen_indices,
                calc=None,
            )

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        if initial_atoms is not None:
            plot_atoms(
                initial_atoms,
                ax=axes[0, 0],
                radii=get_atom_radii(initial_atoms),
            )
            axes[0, 0].set_title("Initial (top view)")

            plot_atoms(
                initial_atoms,
                ax=axes[1, 0],
                radii=get_atom_radii(initial_atoms),
                rotation=("-90x,0y,0z"),
            )
            axes[1, 0].set_title("Initial (side view)")

        plot_atoms(
            final_best_atoms,
            ax=axes[0, 1],
            radii=get_atom_radii(final_best_atoms),
        )
        axes[0, 1].set_title(f"Final best (Gen {last_gen_idx+1}, top view)")

        plot_atoms(
            final_best_atoms,
            ax=axes[1, 1],
            radii=get_atom_radii(final_best_atoms),
            rotation=("-90x,0y,0z"),
        )
        axes[1, 1].set_title(f"Final best (Gen {last_gen_idx+1}, side view)")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Could not create initial vs final comparison: {e}")


def view_best_diffusion(
    dst,
    obj="evaluate_population_with_calc",
    es="CHARLX",
    run_id=0,
    save_movie=False,
    movie_filename="best_diffusion.gif",
):

    def safe_decode(x):
        if x is None:
            return None
        try:
            return decode(x) if isinstance(x, str) else x
        except Exception:
            return x

    h5_files = [
        f
        for f in os.listdir(dst)
        if f.endswith(".h5") and es in f and obj in f
    ]

    if not h5_files:
        h5_files = [f for f in os.listdir(dst) if f.endswith(".h5")]

    if not h5_files:
        print(f"No H5 file found in {dst}")
        return

    h5_path = os.path.join(dst, h5_files[0])
    print(f"Loading data from: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        run_key = f"run_{run_id}"
        if run_key not in f:
            print(
                f"Run ID {run_id} not found in HDF5. Available runs: {list(f.keys())}"
            )
            return

        grp = f[run_key]

        if "objective" not in grp.attrs:
            print("Error: 'objective' attribute missing in HDF5.")
            return

        obj_json = grp.attrs["objective"]

        try:
            obj_dict = json.loads(obj_json)
            obj_params = obj_dict["foo_kwargs"]["obj_params"]
        except Exception as e:
            print(f"Error parsing objective parameters: {e}")
            return

        if "final_best_diffusion" in grp:
            diff_data = grp["final_best_diffusion"][()]
            print(f"Found trajectory with {len(diff_data)} steps.")
        else:
            print("Dataset 'final_best_diffusion' not found.")
            print(
                "Did you run with --save_diffusion and ensure 'delete_folder' was False?"
            )
            return

    founder_atoms = safe_decode(obj_params.get("founder_atoms"))
    fixed_atoms = safe_decode(obj_params.get("fixed_atoms"))
    free_atoms = safe_decode(obj_params.get("free_atoms"))
    frozen_atoms = safe_decode(obj_params.get("frozen_atoms", None))

    fixed_indices = obj_params.get("fixed_indices", None)
    frozen_indices = obj_params.get("frozen_indices", None)
    free_indices = obj_params.get("free_indices", None)

    print("Reconstructing frames...")
    diffusion_atoms = []

    skip = 1
    # if len(diff_data) > 500:
    #     skip = 5

    for i in range(0, len(diff_data), skip):
        step_pos = diff_data[i]

        atoms = combine_fixed_frozen_and_free_atoms(
            founder_atoms=founder_atoms,
            fixed_atoms=fixed_atoms,
            fixed_indices=fixed_indices,
            free_atoms=sample_to_atoms(step_pos, free_atoms, free_indices),
            frozen_atoms=frozen_atoms,
            frozen_indices=frozen_indices,
            calc=None,
        )
        diffusion_atoms.append(atoms)

    if save_movie:
        out_path = os.path.join(dst, movie_filename)
        write(out_path, diffusion_atoms)
        print(f"Saved movie to {out_path}")
    else:
        view(diffusion_atoms)
