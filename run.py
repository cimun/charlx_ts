"""
run.py

Implementation of evolution execution functionalities.
"""

import os
from functools import partial

import h5py
import numpy as np
import torch
from foobench import Objective
from foobench.objective import apply_limits

from ggddim import GGDDIM

print = partial(print, flush=True)

H5_FILE = "ES_{ES}-objective_{objective}.h5"


class CorrectedApplyLimitsObjective(Objective):

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # check limits
        if isinstance(self.foo, type):
            foo = self.foo(**self.foo_kwargs)
        else:
            foo = partial(self.foo, **self.foo_kwargs)

        if not self.maximize:
            f = foo(x)
        else:
            f = -foo(x)

        ### Corrected by changing apply_limits to self.apply_limits
        if not self.apply_limits:
            return f

        return apply_limits(f, x, val=self.limit_val, limits=self.limits)


def to_json(obj):
    """
    convert object to json
    """
    import json

    try:
        return json.dumps(obj)
    except TypeError:
        if not isinstance(obj, dict):
            obj = obj.to_dict() if hasattr(obj, "to_dict") else obj.__dict__
        return json.dumps({k: repr(v) for k, v in obj.items()})


def load_file(file):
    """
    load file as json or yaml
    """
    import os

    if os.path.exists(file):
        if file.endswith(".yml"):
            import yaml

            with open(file, "r") as f:
                return yaml.load(f, Loader=yaml.FullLoader)

        elif file.endswith(".json"):
            import json

            with open(file, "r") as f:
                return json.load(f)

    else:
        # inspect whether file is a json-string
        import json

        try:
            return json.loads(file)
        except json.JSONDecodeError:
            pass

    raise FileNotFoundError(f"File `{file}` not found, or not supported")


def load_nn(nn_cls, config, num_params):
    """
    load `condevo.nn.<nn_cls>` neural network instance  with `config`
    specifying the corresponding constructor `kwargs`, and `num_params`
    specifying the input size.

    :param nn_cls: str or class, neural network class.
    :param config: dict or str, configuration of the neural network, can also
        be a file path or json-string.
    :param num_params: int, input size of the neural network (extra to config,
        as this might depend on the objective function)

    :return: tuple of (nn_instance, config-dict)
    """
    if config is None:
        import configs

        config = getattr(
            configs, nn_cls if isinstance(nn_cls, str) else nn_cls.__name__
        )

    elif isinstance(config, str):
        config = load_file(config)

    assert isinstance(config, dict), "nn_config should be a dictionary"

    if "num_params" not in config:
        config["num_params"] = num_params

    # load nn
    if isinstance(nn_cls, str):
        from condevo import nn

        nn_cls = getattr(nn, nn_cls)

    nn_instance = nn_cls(**config)
    return nn_instance, config


def load_diffuser(diff_cls, config, nn_instance):
    """
    load `condevo.diffusion.<diff_cls>` diffusion instance with `config`
    specifying the corresponding constructor `kwargs`, and `nn_instance`
    specifying the neural network instance.

    :param diff_cls: str or class, diffusion class
    :param config: dict or str, configuration of the diffusion model, can also
        be a file path or json-string.
    :param nn_instance: condevo.nn.<nn_cls>, neural network instance

    :return: tuple of (diff_instance, config-dict)
    """
    if config is None:
        import configs

        config = getattr(
            configs,
            diff_cls if isinstance(diff_cls, str) else diff_cls.__name__,
        )

    elif isinstance(config, str):
        config = load_file(config)

    assert isinstance(config, dict), "diff_config should be a dictionary"

    if diff_cls == "GGDDIM":
        diff_instance = GGDDIM(nn=nn_instance, **config)
    else:
        # load diffuser
        if isinstance(diff_cls, str):
            from condevo import diffusion

            diff_cls = getattr(diffusion, diff_cls)
        diff_instance = diff_cls(nn=nn_instance, **config)

    return diff_instance, config


def load_es(es_cls, config, diffuser, num_params):
    """
    load `condevo.es.<es_cls>` evolutionary strategy instance with `config`
    specifying the corresponding constructor `kwargs`, and `diffuser`
    specifying the diffusion model instance.

    :param es_cls: str or class, `condevo` evolutionary strategy class
    :param config: dict or str, configuration of the evolutionary strategy, can
        also be a file path or json-string.
    :param diffuser: (Optional) condevo.diffusion.<diff_cls>, diffusion model
        instance. If provided, the model will be passed to the evolutionary
        strategy constructor.
    :param num_params: int, number of parameters of the evolutionary search,
        which should be equal to input size of the neural network if provided
        (extra to config, as this might depend on the objective function).

    :return: tuple of (es_instance, config-dict)
    """

    if config is None:
        import configs

        config = getattr(
            configs, es_cls if isinstance(es_cls, str) else es_cls.__name__
        )

    elif isinstance(config, str):
        config = load_file(config)

    assert isinstance(config, dict), "es_config should be a dictionary"

    # load es
    if isinstance(es_cls, str):
        from condevo import es

        es_cls = getattr(es, es_cls)

    # inspect `es_cls` constructor whether "model" is a parameter
    try:
        es_instance = es_cls(num_params=num_params, model=diffuser, **config)
    except TypeError:
        es_instance = es_cls(num_params=num_params, **config)

    return es_instance, config


def run_evo(
    objective='{"foo": "rastrigin", "limits": 3}',
    generations=20,
    nn="MLP",
    nn_config=None,
    diff="DDIM",
    diff_config=None,
    es="HADES",
    es_config=None,
    dst="output/",
    quiet=False,
    timestamp=False,
    params=None,
) -> tuple:

    if not quiet:
        print(f"# Loading Objective:")
    objective_instance = Objective.load(objective)
    if not quiet:
        print(f"-  {objective}")

    nn_instance, diff_instance = None, None

    if nn is not None and diff is not None:

        if not quiet:
            print(f"# Loading Neural Network:")
            print(f"- {nn}")
        nn_instance, nn_config = load_nn(nn, nn_config, objective_instance.dim)
        if not quiet:
            print(f"- {to_json(nn_config)}")

        if not quiet:
            print(f"# Loading Diffusion Model:")
            print(f"- {diff}")
        diff_instance, diff_config = load_diffuser(
            diff, diff_config, nn_instance
        )
        if not quiet:
            print(f"- {to_json(diff_config)}")

    if not quiet:
        print(f"# Loading Evolutionary Strategy")
        print(f"-  {es}")
    solver, es_config = load_es(
        es, es_config, diff_instance, objective_instance.dim
    )
    if not quiet:
        print(f"- {to_json(es_config)}")

    es_name = es if isinstance(es, str) else es.__name__
    h5_filename = os.path.join(
        dst,
        H5_FILE.format(ES=es_name, objective=objective_instance.foo_name),
    )

    if timestamp:
        if not isinstance(timestamp, str):
            from datetime import datetime

            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:-3]
        h5_filename = h5_filename.replace(".h5", f"-{timestamp}.h5")

    os.makedirs(dst, exist_ok=True)
    with h5py.File(h5_filename, "a") as f:
        runs = f.keys()
        num_run = len(np.unique([str(r) for r in runs]))
        print(
            f"# Results are saved \n- in folder '{dst}'\n- as '{h5_filename}'\n- with run_id '{num_run}'."
        )

        run = f.create_group(f"run_{num_run}")

        run.attrs["objective"] = repr(objective_instance)
        run.attrs["generations"] = generations
        if diff is not None:
            run.attrs["nn"] = nn if isinstance(nn, str) else nn.__name__
            run.attrs["nn_config"] = to_json(nn_config)
            run.attrs["diff"] = (
                diff if isinstance(diff, str) else diff.__name__
            )
            run.attrs["diff_config"] = to_json(diff_config)
        run.attrs["es"] = es if isinstance(es, str) else es.__name__
        run.attrs["es_config"] = to_json(es_config)
        run.attrs["dst"] = dst
        run.attrs["quiet"] = quiet

        try:
            if hasattr(objective_instance, "foo_kwargs") and isinstance(
                objective_instance.foo_kwargs, dict
            ):
                objp = objective_instance.foo_kwargs.get("obj_params", None)
                if isinstance(objp, dict):
                    objp["h5_filename"] = h5_filename
                    objp["run_id"] = num_run
                    objective_instance.foo_kwargs["obj_params"] = objp
        except Exception:
            pass

    i, fitness, model_loss = 0, [], [0]

    should_save_diff = params.get("save_diffusion", False) if params else False
    stored_best_traj = None
    stored_best_fitness = -float("inf")

    for i in range(generations):
        # We must capture every generation because we don't know if a new best will appear.
        if hasattr(solver, "model"):
            solver.model.save_diffusion_flag = should_save_diff

        samples = solver.ask()
        current_gen_diffusion_data = None

        if (
            should_save_diff
            and hasattr(solver, "model")
            and hasattr(solver.model, "latest_diffusion")
        ):
            current_gen_diffusion_data = solver.model.latest_diffusion.clone()
            del solver.model.latest_diffusion
            solver.model.save_diffusion_flag = False

        if hasattr(solver, "relaxed_atoms_list"):
            try:
                objective_instance.foo_kwargs["obj_params"][
                    "relaxed_atoms_list"
                ] = solver.relaxed_atoms_list
            except Exception:
                pass

        fitness = objective_instance(samples)
        model_loss = solver.tell(fitness)

        if should_save_diff:
            # Get best of current generation
            fit_np = (
                fitness.cpu().numpy()
                if isinstance(fitness, torch.Tensor)
                else fitness
            )
            curr_max_fit = np.max(fit_np)
            curr_best_idx = np.argmax(fit_np)

            num_elites = getattr(solver, "num_elite", 0)

            if curr_max_fit > stored_best_fitness:

                # Check if this record holder is a NEW offspring or an OLD Elite
                if curr_best_idx >= num_elites:
                    # It is a NEW offspring. We must grab its trajectory.
                    if current_gen_diffusion_data is not None:
                        traj_idx = curr_best_idx - num_elites

                        if traj_idx < current_gen_diffusion_data.shape[0]:
                            stored_best_traj = current_gen_diffusion_data[
                                traj_idx
                            ].clone()
                            stored_best_fitness = curr_max_fit
                            print(
                                f">> New Global Best found in Gen {i} (Index {curr_best_idx}). Updating stored trajectory."
                            )
                        else:
                            print(
                                f"Warning: New best index {traj_idx} out of bounds for diffusion data."
                            )

                else:
                    # It is an ELITE (Survivor).
                    stored_best_fitness = curr_max_fit
                    print(
                        f">> Global Best in Gen {i} is an Elite (Index {curr_best_idx}). Keeping previous trajectory."
                    )

        current_gen_diffusion_data = None

        # Refresh samples for saving in HDF5 output file
        ### samples = solver.solutions

        if not quiet and diff is not None:
            print(
                "  {"
                + f' "Generation": {i}, "Max-Fitness": {fitness.max()}, "Avg-Fitness": {fitness.mean()}, "Model-Loss": {model_loss[-1]}'
                + "}"
            )
        elif not quiet:
            print(
                "  {"
                + f' "Generation": {i}, "Max-Fitness": {fitness.max()}, "Avg-Fitness": {fitness.mean()}'
                + "}"
            )

        samples = (
            samples.numpy() if not isinstance(samples, np.ndarray) else samples
        )
        fitness = (
            fitness.numpy() if not isinstance(fitness, np.ndarray) else fitness
        )
        model_loss = np.array(model_loss)

        with h5py.File(h5_filename, "a") as f:
            f.create_dataset(f"run_{num_run}/gen_{i}/samples", data=samples)
            f.create_dataset(f"run_{num_run}/gen_{i}/fitness", data=fitness)
            if diff is not None:
                f.create_dataset(
                    f"run_{num_run}/gen_{i}/model_loss", data=model_loss
                )

            if hasattr(solver, "relaxed_atoms_list"):
                relaxed_positions = np.array(
                    [
                        atoms.get_positions()
                        for atoms in solver.relaxed_atoms_list
                    ]
                )
                f.create_dataset(
                    f"run_{num_run}/gen_{i}/relaxed_positions",
                    data=relaxed_positions,
                )

    # Save diffusion trajectory
    if stored_best_traj is not None:
        print(
            f"Saving final best diffusion trajectory (Fitness: {stored_best_fitness})"
        )

        best_diff = stored_best_traj.cpu().numpy()

        with h5py.File(h5_filename, "a") as f:
            dset_name = f"run_{num_run}/final_best_diffusion"
            if dset_name in f:
                del f[dset_name]
            f.create_dataset(dset_name, data=best_diff)

    if not quiet and diff is not None:
        print(
            "  {"
            + f' "Generation": {i}, "Max-Fitness": {fitness.max()}, "Avg-Fitness": {fitness.mean()}, "Model-Loss": {model_loss[-1]}'
            + "}"
        )
    elif not quiet:
        print(
            "  {"
            + f' "Generation": {i}, "Max-Fitness": {fitness.max()}, "Avg-Fitness": {fitness.mean()}'
            + "}"
        )

    return h5_filename, num_run, solver.result()[0], solver.result()[1]
