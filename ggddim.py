"""
ggddim.py

Implementation of the GGDM and GGDDIM classes.
"""

from functools import partial
import inspect

import numpy as np
import torch
from condevo.diffusion import DM
from torch import (
    Tensor,
    cat,
    clamp,
    cos,
    linspace,
    no_grad,
    norm,
    ones,
    ones_like,
    pi,
    rand,
    randn_like,
    sqrt,
    tensor,
    triu,
    vmap,
    zeros,
    zeros_like,
)
from torch.nn import ReLU


class GGDM(DM):
    """Gradient guidance diffusion model base-class."""

    @no_grad()
    def sample(
        self,
        shape: tuple,
        num: int = None,
        x_source: Tensor = None,
        conditions=None,
        t_start=None,
        **kwargs,
    ):
        """
        Sample `num` points from the diffusion model with given `shape` and `conditions`. If `num` is not specified, sample points from `x_source` tensor.

        Args:
            shape (tuple): Shape of the sampled points.
            num (int, optional): Number of points to be sampled. Defaults to None.
            x_source (torch.tensor, optional): Source tensor to sample points. Defaults to None.
            conditions (tuple, optional): Conditions for the diffusion model. Defaults to None.
            t_start (int, optional): Starting time step for the diffusion process. Defaults to None.

        Returns:
            torch.tensor: Sampled points from the diffusion model.
        """

        if (num is None) and (x_source is None):
            raise ValueError("Either `num` or `x_source` should be specified")

        if (num is not None) and (x_source is not None):
            raise ValueError(
                "Only one of `num` and `x_source` should be specified"
            )

        if conditions is None:
            conditions = tuple()

        if num is not None:
            x_source = self.draw_random(num, *shape)

        self.eval()
        self.nn.eval()

        do_save = kwargs.get("save_diffusion", False) or getattr(
            self, "save_diffusion_flag", False
        )

        sample_vectorized = vmap(
            partial(self.sample_point, save_diffusion=do_save),
            randomness="different",
        )
        x_sampled = sample_vectorized(x_source, *conditions, t_start=t_start)

        if do_save:
            self.latest_diffusion = x_sampled.clone()

            # Return only the LAST step (Batch, Dim) so standard ES logic continues working
            x_sampled = x_sampled[:, -1, :]
            return x_sampled

        # check for valid parameter range
        exceeding_x = self.exceeds_diff_range(x_sampled) > 0
        exceeding_count = 0
        while self.diff_range not in [None, 0, 0.0] and exceeding_x.any():
            # new sample points
            exceeding_x_source = self.draw_random(
                int(sum(exceeding_x)), *shape
            )

            if exceeding_count > 2:  # try 10 times to sample valid points
                # clamp to diff_range if too many iterations
                exceeding_x_source = self.diff_clamp(exceeding_x_source)
                x_sampled[exceeding_x] = exceeding_x_source
                break

            else:
                exceeding_conditions = [
                    condition[exceeding_x] for condition in conditions
                ]
                x_resampled = sample_vectorized(
                    exceeding_x_source, *exceeding_conditions
                )

                # check for valid parameter range and integrate into samples
                exceeding_resampled = self.exceeds_diff_range(x_resampled) > 0
                valid_resampled = torch.where(exceeding_x)[0][
                    ~exceeding_resampled
                ]
                x_sampled[valid_resampled] = x_resampled[~exceeding_resampled]

            exceeding_x = self.exceeds_diff_range(x_sampled) > 0
            exceeding_count += 1

        return x_sampled


class GGDDIM(GGDM):
    """
    GGDDIM: Gradient Guided Diffusion with Virtual Origin support.

    Integrated with VirtualOriginGuidance logic for:
    - Coordinate transformations (Global <-> Local)
    - Extended Geometries (Box, Sphere, Axis, Ellipsoid, Half-Sphere)
    """

    ALPHA_SCHEDULES = ["linear", "cosine"]

    def __init__(
        self,
        nn,
        num_steps=1000,
        skip_connection=True,
        noise_level=1.0,
        diff_range=None,
        lambda_range=0.0,
        predict_eps_t=False,
        param_mean=0.0,
        param_std=1.0,
        alpha_schedule="linear",
        matthew_factor=1.0,
        sample_uniform=False,
        autoscaling=False,
        log_dir="",
        normalize_steps=False,
        diff_range_filter=True,
        clip_gradients=None,
        geometry="radial",
        axis=2,  # z-axis
        lower_threshold=0.0,
        upper_threshold=3.0,
        min_distance=1.0,
        overlap_penalty=True,
        power=2.0,
        scale=50.0,
        max_grad_norm=1.0,
        schedule_type="constant",
        scale_base=1.0,
        gradient_interval=1,
        train_on_penalty=True,
        diff_origin=[0.0, 0.0, 0.0],
        **kwargs,
    ):

        dm_kwargs = dict(
            nn=nn,
            num_steps=num_steps,
            diff_range=diff_range,
            lambda_range=lambda_range,
            log_dir=log_dir,
            diff_range_filter=diff_range_filter,
            clip_gradients=clip_gradients,
            **kwargs,
        )
        dm_sig = inspect.signature(DM.__init__)
        dm_params = dm_sig.parameters
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in dm_params.values()
        )
        if not accepts_var_kwargs:
            dm_kwargs = {k: v for k, v in dm_kwargs.items() if k in dm_params}

        super(GGDDIM, self).__init__(**dm_kwargs)

        # Backward-compatibility for legacy configs/behavior.
        self.param_mean = param_mean
        self.param_std = param_std
        self.sample_uniform = sample_uniform
        self.autoscaling = autoscaling

        self.skip_connection = skip_connection
        self.normalize_steps = normalize_steps

        self.geometry = geometry
        self.axis = axis
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.min_distance = min_distance
        self.overlap_penalty = overlap_penalty
        self.power = power
        self.scale = scale
        self.max_grad_norm = max_grad_norm
        self.schedule_type = schedule_type
        self.scale_base = scale_base
        self.gradient_interval = gradient_interval
        self.train_on_penalty = train_on_penalty

        origin = diff_origin
        self.register_buffer("origin", tensor(origin, device=self.device))

        self.noise_level = noise_level
        self.predict_eps_t = predict_eps_t
        self._alpha_schedule = None
        self.matthew_factor = matthew_factor

        self.alpha_schedule = alpha_schedule

    @property
    def alpha_schedule(self):
        return self._alpha_schedule

    @alpha_schedule.setter
    def alpha_schedule(self, value):
        if value not in self.ALPHA_SCHEDULES:
            raise ValueError(f"Invalid alpha schedule: {value}")
        self._alpha_schedule = value

        if value == "linear":
            alpha_tensor = linspace(
                1 - 1 / self.num_steps, 1e-8, self.num_steps
            ).to(self.device)
        elif value == "cosine":
            delta = 1e-3
            x = linspace(0, pi, self.num_steps)
            alpha_tensor = ((cos(x) * (1 - 2 * delta) + 1) / 2).to(self.device)

        a = cat([tensor([1], device=self.device), alpha_tensor])
        sigma_tensor = (1 - a[:-1]) / (1 - a[1:]) * (1 - a[1:] / a[:-1])
        sigma_tensor = sqrt(sigma_tensor)

        self.register_buffer("alpha", alpha_tensor)
        self.register_buffer("sigma", sigma_tensor)

    # Virtual Origin
    def _get_origin_flat(self, x_shape):
        """Broadcasts origin to match flattened batch shape (Batch, N*3)."""
        dim = x_shape[-1]
        if dim > 3 and dim % 3 == 0:
            n_atoms = dim // 3
            return self.origin.repeat(n_atoms).unsqueeze(0)
        return self.origin.unsqueeze(0)

    def global_to_local(self, x_global):
        """Shifts global coordinates to local diffusion frame (centered at 0)."""
        return x_global - self._get_origin_flat(x_global.shape)

    def local_to_global(self, x_local):
        """Shifts local diffusion coordinates back to global frame."""
        return x_local + self._get_origin_flat(x_local.shape)

    def forward(self, xt, t, *conditions):
        if self.normalize_steps:
            t = t / self.num_steps
        y = super().forward(xt.clone(), t, *conditions)
        return y + xt if self.skip_connection else y

    def diffuse(self, x, t):
        """Adds noise to data x (Assumes x is already LOCAL)."""
        eps = randn_like(x)
        if self.autoscaling:
            eps = eps * self.param_std

        if isinstance(t, float):
            t = tensor(t, device=x.device)

        T = (t * (self.num_steps - 1)).long()

        eps_t = (1 - self.alpha[T]).sqrt() * eps
        xt = self.alpha[T].sqrt() * x + eps_t

        if self.predict_eps_t:
            eps_t = (xt - self.alpha[T].sqrt() * x) / (
                1 - self.alpha[T]
            ).sqrt()
            return xt, eps_t
        return xt, eps

    # Gradient Guidance - operates in local frame

    def _get_geometry_params(self, x_device):
        local_center = tensor([0.0, 0.0, 0.0], device=x_device)

        local_lower = None
        local_upper = None

        if self.geometry in ["axis", "box"]:
            if self.lower_threshold is not None:
                l_ten = tensor(self.lower_threshold, device=x_device)
                if self.geometry == "axis":
                    local_lower = l_ten - self.origin[self.axis]
                else:
                    local_lower = l_ten - self.origin

            if self.upper_threshold is not None:
                u_ten = tensor(self.upper_threshold, device=x_device)
                if self.geometry == "axis":
                    local_upper = u_ten - self.origin[self.axis]
                else:
                    local_upper = u_ten - self.origin

        elif self.geometry in ["radial", "ellipsoid", "half_sphere"]:
            if self.lower_threshold is not None:
                local_lower = tensor(self.lower_threshold, device=x_device)
            if self.upper_threshold is not None:
                local_upper = tensor(self.upper_threshold, device=x_device)

        return local_center, self.power, local_lower, local_upper

    def _calculate_penalty(self, x_vec):
        """Geometric penalty for Loss Function."""
        if x_vec.dim() == 2 and x_vec.size(0) == 1:
            x = x_vec.squeeze(0)
        else:
            x = x_vec

        coords = x.view(-1, 3)
        if not self.geometry:
            return tensor(0.0, device=x.device, dtype=coords.dtype)

        center, power, lower, upper = self._get_geometry_params(x.device)

        violation = tensor(0.0, device=x.device)

        if (
            self.geometry == "radial"
            and lower is not None
            and upper is not None
        ):
            d = norm(coords - center, dim=1)
            below = clamp(lower - d, min=0.0)
            above = clamp(d - upper, min=0.0)
            violation = (below + above) ** power

        elif (
            self.geometry == "axis" and lower is not None and upper is not None
        ):
            vals = coords[:, self.axis]
            below = clamp(lower - vals, min=0.0)
            above = clamp(vals - upper, min=0.0)
            violation = (below + above) ** power

        elif self.geometry == "box":
            l_ten = (
                lower
                if lower is not None
                else tensor([-1e9] * 3, device=x.device)
            )
            u_ten = (
                upper
                if upper is not None
                else tensor([1e9] * 3, device=x.device)
            )
            below = clamp(l_ten - coords, min=0.0)
            above = clamp(coords - u_ten, min=0.0)
            violation = (below + above).sum(dim=1) ** power

        elif self.geometry == "ellipsoid" and upper is not None:
            radii = (
                upper
                if isinstance(upper, torch.Tensor)
                else tensor([upper] * 3, device=x.device)
            )
            normalized_diff = (coords - center) / radii
            dist_sq = (normalized_diff**2).sum(dim=1)
            d_ellip = sqrt(dist_sq + 1e-12)
            above = clamp(d_ellip - 1.0, min=0.0)
            violation = above**power

        elif (
            self.geometry == "half_sphere"
            and lower is not None
            and upper is not None
        ):
            d = norm(coords - center, dim=1)
            below_rad = clamp(lower - d, min=0.0)
            above_rad = clamp(d - upper, min=0.0)
            vals = coords[:, self.axis] - center[self.axis]
            plane_violation = clamp(-vals, min=0.0)
            violation = (below_rad + above_rad + plane_violation) ** power

        total = violation.sum()

        if self.overlap_penalty:
            if self.min_distance > 0:
                diff = coords[:, None, :] - coords[None, :, :]
                dmat = sqrt((diff * diff).sum(dim=2) + 1e-12)
                mask = triu(ones_like(dmat), diagonal=1)
                overlaps = clamp(
                    tensor(self.min_distance, device=dmat.device) - dmat,
                    min=0.0,
                )
                total = total + (overlaps**2 * mask).sum()

        return total

    def _manual_gradient(self, x_vec):
        """
        Analytical gradient for guidance.
        """
        grads = zeros_like(x_vec)
        coords = x_vec.view(-1, 3)
        center, power, lower_val, upper_val = self._get_geometry_params(
            x_vec.device
        )

        def compute_pow_grad(dist, target, sign_direction):
            diff = dist - target
            is_active = (diff * sign_direction > 0).float()
            mag = (
                sign_direction
                * power
                * (diff.abs().clamp(min=1e-8) ** (power - 1))
            )
            return is_active * mag

        # Geometry gradients
        if lower_val is not None or upper_val is not None:
            if self.geometry == "radial":
                diff_vec = coords - center
                d = norm(diff_vec, dim=1) + 1e-12
                direction = diff_vec / d.unsqueeze(-1)
                mag = torch.zeros_like(d)
                if lower_val is not None:
                    mag += compute_pow_grad(d, lower_val, -1)
                if upper_val is not None:
                    mag += compute_pow_grad(d, upper_val, 1)
                grads.view(-1, 3)[:, :] += direction * mag.unsqueeze(-1)

            elif self.geometry == "axis":
                vals = coords[:, self.axis]
                grad_val = torch.zeros_like(vals)
                if lower_val is not None:
                    grad_val += compute_pow_grad(vals, lower_val, -1)
                if upper_val is not None:
                    grad_val += compute_pow_grad(vals, upper_val, 1)
                grads.view(-1, 3)[:, self.axis] += grad_val

            elif self.geometry == "box":
                for i in range(3):
                    vals = coords[:, i]
                    if lower_val is not None:
                        grads.view(-1, 3)[:, i] += compute_pow_grad(
                            vals, lower_val[i], -1
                        )
                    if upper_val is not None:
                        grads.view(-1, 3)[:, i] += compute_pow_grad(
                            vals, upper_val[i], 1
                        )

            elif self.geometry == "ellipsoid":
                diff_vec = coords - center
                radii = (
                    upper_val
                    if isinstance(upper_val, torch.Tensor)
                    else tensor([upper_val] * 3, device=x_vec.device)
                )
                normalized_vec = diff_vec / radii
                d_ellip = norm(normalized_vec, dim=1) + 1e-12
                mag = compute_pow_grad(
                    d_ellip, tensor(1.0, device=x_vec.device), 1
                )
                direction = diff_vec / (radii**2 * d_ellip.unsqueeze(-1))
                grads.view(-1, 3)[:, :] += direction * mag.unsqueeze(-1)

            elif self.geometry == "half_sphere":
                # Radial part
                diff_vec = coords - center
                d = norm(diff_vec, dim=1) + 1e-12
                direction = diff_vec / d.unsqueeze(-1)
                mag_rad = torch.zeros_like(d)
                if lower_val is not None:
                    mag_rad += compute_pow_grad(d, lower_val, -1)
                if upper_val is not None:
                    mag_rad += compute_pow_grad(d, upper_val, 1)
                grads.view(-1, 3)[:, :] += direction * mag_rad.unsqueeze(-1)

                # Plane part
                vals = coords[:, self.axis] - center[self.axis]
                mag_plane = compute_pow_grad(
                    vals, tensor(0.0, device=x_vec.device), -1
                )
                grads.view(-1, 3)[:, self.axis] += mag_plane

        # Overlap gradient
        if self.overlap_penalty:
            if self.min_distance > 0:
                diff_mat = coords[:, None, :] - coords[None, :, :]
                dmat = norm(diff_mat, dim=2) + 1e-12
                overlaps = self.min_distance - dmat
                is_active = ((overlaps > 0) & (dmat > 1e-8)).float()
                grad_mag = -2 * overlaps
                direction = diff_mat / dmat.unsqueeze(-1)
                pairwise_grads = (
                    direction
                    * grad_mag.unsqueeze(-1)
                    * is_active.unsqueeze(-1)
                )
                total_grads = pairwise_grads.sum(dim=1)
                grads.view(-1, 3)[:, :] += total_grads

        return grads

    def fit(self, x, *conditions, weights=None, **kwargs):
        """Transforms GLOBAL data to LOCAL frame before training."""
        x_local = self.global_to_local(x)
        return super().fit(x_local, *conditions, weights=weights, **kwargs)

    def regularize(self, x_batch, w_batch, *c_batch):
        """Physics-Informed regularization during training (Local Frame)."""
        if self.lambda_range > 0:
            t = rand(x_batch.shape[0], device=self.device).reshape(-1, 1)
            T = (t * (self.num_steps - 1)).long()

            xt, _ = self.diffuse(x_batch, t)
            eps_pred = self(xt, t, *c_batch)

            score = eps_pred / (self.alpha[T].sqrt())
            x0_pred = xt - score

            loss_sphere = self.exceeds_diff_range(x0_pred)[:, None]
            loss_geom = tensor(0.0, device=self.device)
            if self.train_on_penalty and self.geometry:
                loss_geom = self._calculate_penalty(x0_pred) / x_batch.shape[0]

            return self.lambda_range * (loss_sphere.mean() + loss_geom)

        return super(GGDDIM, self).regularize(x_batch, w_batch, *c_batch)

    def eval_val_pred(self, x, *conditions):
        t = rand(x.shape[0], device=self.device).reshape(-1, 1)
        xt, eps = self.diffuse(x, t)
        eps_pred = self.forward(xt, t, *conditions)
        return eps, eps_pred

    def sample_point(
        self, xt, *conditions, t_start=None, save_diffusion=False
    ):
        if t_start is None:
            t_start = self.num_steps - 1
        if t_start == self.num_steps - 1:
            xt = randn_like(xt)

        if xt.dim() == 1:
            xt = xt.unsqueeze(0)

        reshaped_conds = []
        for c in conditions:
            if c.dim() == 1 and xt.shape[0] == 1:
                reshaped_conds.append(c.unsqueeze(0))
            elif c.dim() == 0:
                reshaped_conds.append(c.view(1, 1))
            else:
                reshaped_conds.append(c)
        conditions = tuple(reshaped_conds)

        one = ones(1, 1, device=self.device)
        diffusion_process = []
        x0_process = []

        if save_diffusion:
            diffusion_process.append(
                self.local_to_global(xt).squeeze(0).clone()
            )
            x0_process.append(self.local_to_global(xt).squeeze(0).clone())

        for T in range(t_start - 1, 0, -1):
            t = one * T / self.num_steps
            s = self.sigma[T - 1] * self.noise_level
            z = randn_like(xt)

            eps = self(xt, t, *conditions) * self.matthew_factor

            if self.predict_eps_t:
                x0_pred = xt - eps
            else:
                x0_pred = (xt - (1 - self.alpha[T]).sqrt() * eps) / (
                    self.alpha[T].sqrt()
                )

            t_ratio = T / self.num_steps
            current_scale = self.scale_base
            if self.schedule_type == "linear_increase":
                current_scale = self.scale_base * (1.0 - t_ratio)
            elif self.schedule_type == "linear_decay":
                current_scale = self.scale_base * t_ratio
            elif self.schedule_type == "warmup_cosine":
                current_scale = self.scale_base * (
                    0.5 - 0.5 * np.cos(np.pi * (1 - t_ratio))
                )

            if current_scale != 0.0 and (T % self.gradient_interval == 0):
                grads = self._manual_gradient(x0_pred)

                update = current_scale * grads
                update_norm = norm(update, p=2, dim=-1, keepdim=True)
                ratio = self.max_grad_norm / (update_norm + 1e-8)
                scale_factor = clamp(ratio, max=1.0)

                x0_pred = x0_pred - (update * scale_factor)

            xt = (
                self.alpha[T - 1].sqrt() * x0_pred
                + (1 - self.alpha[T - 1] - s**2).sqrt() * eps
                + s * z
            )

            if save_diffusion:
                diffusion_process.append(
                    self.local_to_global(xt).squeeze(0).clone()
                )
                x0_process.append(
                    self.local_to_global(x0_pred).squeeze(0).clone()
                )

        if save_diffusion:
            return torch.stack(diffusion_process)

        return self.local_to_global(xt).squeeze(0)

    def exceeds_diff_range(self, x):
        if self.diff_range in [None, 0, 0.0]:
            return zeros_like(x[:, 0], device=self.device)
        return (
            ReLU()((x**2 - self.diff_range**2).reshape(x.shape[0], -1))
            .mean(dim=-1)
            .sqrt()
        )
