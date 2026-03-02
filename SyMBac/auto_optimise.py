"""
Automatic parameter optimisation for SyMBac synthetic image matching.

Uses Optuna (Bayesian optimisation with TPE) to find rendering + PSF
parameters that minimise a composite image similarity loss between
synthetic and real microscopy images.
"""

import numpy as np
import optuna
import warnings
from tqdm.auto import tqdm

from SyMBac.PSF import PSF_generator
from SyMBac.metrics import (
    composite_loss,
    intensity_histogram_emd,
    power_spectral_density_error,
)


optuna.logging.set_verbosity(optuna.logging.WARNING)


class AutoOptimiser:
    """
    Automatic parameter optimisation for synthetic image matching.

    Given a Renderer and one or more real microscopy images, finds the
    rendering parameters that produce the most realistic synthetic images.

    Example
    -------
    >>> from SyMBac.auto_optimise import AutoOptimiser
    >>> optimiser = AutoOptimiser(my_renderer, real_images=[real_image])
    >>> best_params = optimiser.optimise()
    >>> optimiser.plot_results()
    >>> optimiser.apply_to_renderer()
    >>> my_renderer.generate_training_data(...)
    """

    def __init__(self, renderer, real_images, n_trials=300, timeout=600,
                 metrics=None, weights=None, n_scenes_per_eval=3,
                 optimise_psf=False, cells="dark"):
        """
        Parameters
        ----------
        renderer : SyMBac.renderer.Renderer
            A configured Renderer object (simulation + PSF + real image).
        real_images : list of 2D numpy arrays
            Real microscopy images to match against.
        n_trials : int
            Maximum number of Optuna trials.
        timeout : int
            Maximum optimisation time in seconds.
        metrics : list of str, optional
            Which metrics to use. Options: "emd", "psd", "glcm",
            "contrast", "moments". Defaults to all.
        weights : dict, optional
            Per-metric weights. Defaults to equal weighting.
        n_scenes_per_eval : int
            Number of different scene_no values to render and average per
            trial. Higher = less noisy but slower.
        optimise_psf : bool
            If True, include PSF parameters (NA, wavelength, condenser,
            apo_sigma) in the optimisation.
        cells : str
            "dark" if cells are darker than media in phase contrast,
            "light" otherwise. Used for auto-segmentation.
        """
        self.renderer = renderer
        self.real_images = real_images if isinstance(real_images, list) else [real_images]
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_scenes_per_eval = n_scenes_per_eval
        self.optimise_psf = optimise_psf
        self.cells = cells

        if metrics is None:
            self.metrics = ["emd", "psd", "glcm", "contrast", "moments"]
        else:
            self.metrics = metrics

        if weights is None:
            self.weights = {m: 1.0 for m in self.metrics}
        else:
            self.weights = weights

        self.study = None
        self.best_params = None
        self._real_region_stats = None

        # Pre-compute real image statistics for contrast ratio metric
        self._precompute_real_stats()

    def _precompute_real_stats(self):
        """Extract region statistics from real images using auto-segmentation."""
        region_masks = self.renderer.auto_segment_regions(
            image=self.real_images[0], cells=self.cells
        )
        real = self.real_images[0]
        media_pixels = real[region_masks["media"]]
        cell_pixels = real[region_masks["cell"]]
        device_pixels = real[region_masks["device"]]

        self._real_region_stats = {
            "media": float(media_pixels.mean()) if media_pixels.size > 0 else 0.0,
            "cell": float(cell_pixels.mean()) if cell_pixels.size > 0 else 0.0,
            "device": float(device_pixels.mean()) if device_pixels.size > 0 else 0.0,
        }

    def _smart_init(self):
        """
        Extract initial parameter estimates from real images to narrow
        the search space.

        Returns
        -------
        dict
            Initial parameter estimates with narrowed bounds.
        """
        real = self.real_images[0]
        stats = self._real_region_stats

        # Estimate multiplier ratios from real image region means
        # Media is typically brightest in phase contrast
        media_est = stats["media"] * 300  # Scale to multiplier range
        device_est = stats["device"] * 300
        cell_contrast = stats["cell"] / max(stats["media"], 1e-10)

        return {
            "media_multiplier_init": np.clip(media_est, -300, 300),
            "device_multiplier_init": np.clip(device_est, -300, 300),
            "cell_multiplier_init": np.clip(cell_contrast * 5, -30, 30),
        }

    def _define_search_space(self, trial):
        """
        Define the Optuna search space for a trial.

        Parameters
        ----------
        trial : optuna.Trial

        Returns
        -------
        dict
            Parameter dictionary to pass to render_synthetic.
        """
        psf = self.renderer.PSF
        min_sigma = psf.min_sigma

        params = {
            "media_multiplier": trial.suggest_float("media_multiplier", -300, 300),
            "cell_multiplier": trial.suggest_float("cell_multiplier", -30, 30),
            "device_multiplier": trial.suggest_float("device_multiplier", -300, 300),
            "sigma": trial.suggest_float("sigma", min_sigma, min_sigma * 20),
            "noise_var": trial.suggest_float("noise_var", 0, 0.01),
            "defocus": trial.suggest_float("defocus", 0, 20),
            "halo_top_intensity": trial.suggest_float("halo_top_intensity", 0, 1),
            "halo_bottom_intensity": trial.suggest_float("halo_bottom_intensity", 0, 1),
            "halo_start": trial.suggest_float("halo_start", 0, 1),
            "halo_end": trial.suggest_float("halo_end", 0, 1),
            "match_fourier": trial.suggest_categorical("match_fourier", [True, False]),
            "match_histogram": trial.suggest_categorical("match_histogram", [True, False]),
            "match_noise": trial.suggest_categorical("match_noise", [True, False]),
        }

        if self.optimise_psf:
            params["_psf_NA"] = trial.suggest_float("NA", 0.3, 1.5)
            params["_psf_wavelength"] = trial.suggest_float("wavelength", 0.4, 0.8)
            if self.renderer.PSF.mode == "phase contrast":
                params["_psf_condenser"] = trial.suggest_categorical(
                    "condenser", ["Ph1", "Ph2", "Ph3", "Ph4", "PhF"]
                )

        return params

    def _apply_psf_params(self, params):
        """
        If optimising PSF params, create a new PSF with the trial's values.
        """
        if not self.optimise_psf:
            return

        psf = self.renderer.PSF
        new_NA = params.pop("_psf_NA", psf.NA)
        new_wavelength = params.pop("_psf_wavelength", psf.wavelength)
        new_condenser = params.pop("_psf_condenser", psf.condenser)

        self.renderer.PSF = PSF_generator(
            radius=psf.radius,
            wavelength=new_wavelength,
            NA=new_NA,
            n=psf.n,
            resize_amount=self.renderer.simulation.resize_amount,
            pix_mic_conv=self.renderer.simulation.pix_mic_conv,
            apo_sigma=psf.apo_sigma,
            mode=psf.mode,
            condenser=new_condenser,
        )
        self.renderer.PSF.calculate_PSF()

    def _compute_objective(self, params):
        """
        Render synthetic images and compute composite loss.

        Renders multiple scenes and averages the loss to smooth noise.

        Parameters
        ----------
        params : dict
            Rendering parameters from _define_search_space.

        Returns
        -------
        float
            Composite loss value (lower is better).
        """
        # Separate PSF params before rendering
        render_params = {k: v for k, v in params.items() if not k.startswith("_psf_")}
        self._apply_psf_params(params)

        sim_length = len(self.renderer.simulation.OPL_scenes)
        # Pick diverse scene indices
        burn_in = max(1, sim_length // 4)
        scene_indices = np.linspace(
            burn_in, sim_length - 1, self.n_scenes_per_eval, dtype=int
        )

        synth_images = []
        for scene_no in scene_indices:
            try:
                render_params["scene_no"] = int(scene_no)
                noisy_img, _ = self.renderer.render_synthetic(**render_params)
                synth_images.append(noisy_img)
            except Exception:
                return float("inf")

        if not synth_images:
            return float("inf")

        # Pick real images to compare against (cycle if fewer than synth)
        real_for_comparison = []
        for i in range(len(synth_images)):
            real_for_comparison.append(self.real_images[i % len(self.real_images)])

        # Compute synth region stats for contrast ratio metric
        synth_regions = None
        if "contrast" in self.metrics:
            synth_regions = self._estimate_synth_regions(synth_images[0])

        try:
            loss = composite_loss(
                real_images=real_for_comparison,
                synth_images=synth_images,
                weights=self.weights,
                real_regions=self._real_region_stats,
                synth_regions=synth_regions,
            )
        except Exception:
            return float("inf")

        return loss

    def _estimate_synth_regions(self, synth_image):
        """Estimate region mean intensities from a synthetic image."""
        try:
            masks = self.renderer.auto_segment_regions(
                image=synth_image, cells=self.cells
            )
            media_px = synth_image[masks["media"]]
            cell_px = synth_image[masks["cell"]]
            device_px = synth_image[masks["device"]]
            return {
                "media": float(media_px.mean()) if media_px.size > 0 else 0.0,
                "cell": float(cell_px.mean()) if cell_px.size > 0 else 0.0,
                "device": float(device_px.mean()) if device_px.size > 0 else 0.0,
            }
        except Exception:
            return {"media": 0.5, "cell": 0.3, "device": 0.1}

    def optimise(self, show_progress=True):
        """
        Run the full optimisation pipeline.

        Phase 1: TPE-based Bayesian search (main exploration)
        Phase 2: CMA-ES local refinement around the best found parameters

        Parameters
        ----------
        show_progress : bool
            Show a progress bar during optimisation.

        Returns
        -------
        dict
            Best parameter dictionary found.
        """
        # Phase 1: TPE search
        sampler = optuna.samplers.TPESampler(
            multivariate=True, seed=42
        )
        self.study = optuna.create_study(
            direction="minimize", sampler=sampler
        )

        # Seed with smart initialization
        init_params = self._smart_init()
        min_sigma = self.renderer.PSF.min_sigma
        seed_params = {
            "media_multiplier": init_params["media_multiplier_init"],
            "cell_multiplier": init_params["cell_multiplier_init"],
            "device_multiplier": init_params["device_multiplier_init"],
            "sigma": min_sigma * 5,
            "noise_var": 0.001,
            "defocus": 3.0,
            "halo_top_intensity": 1.0,
            "halo_bottom_intensity": 1.0,
            "halo_start": 0.0,
            "halo_end": 1.0,
            "match_fourier": False,
            "match_histogram": True,
            "match_noise": False,
        }
        if self.optimise_psf:
            seed_params["NA"] = self.renderer.PSF.NA
            seed_params["wavelength"] = self.renderer.PSF.wavelength
            if self.renderer.PSF.mode == "phase contrast":
                seed_params["condenser"] = self.renderer.PSF.condenser
        self.study.enqueue_trial(seed_params)

        # Reserve some budget for refinement
        tpe_trials = max(1, int(self.n_trials * 0.8))
        cma_trials = self.n_trials - tpe_trials

        if show_progress:
            pbar = tqdm(total=tpe_trials, desc="Phase 1: TPE search")

            def _tpe_callback(study, trial):
                pbar.update(1)
                pbar.set_postfix({"best_loss": f"{study.best_value:.4f}"})
        else:
            _tpe_callback = None

        self.study.optimize(
            self._objective_wrapper,
            n_trials=tpe_trials,
            timeout=int(self.timeout * 0.8),
            callbacks=[_tpe_callback] if _tpe_callback else None,
        )

        if show_progress:
            pbar.close()

        # Phase 2: CMA-ES refinement (continuous params only)
        if cma_trials > 0 and len(self.study.trials) > 0:
            best_trial = self.study.best_trial
            cma_sampler = optuna.samplers.CmaEsSampler(
                seed=42,
                source_trials=self.study.trials,
            )
            cma_study = optuna.create_study(
                direction="minimize", sampler=cma_sampler
            )

            if show_progress:
                pbar2 = tqdm(total=cma_trials, desc="Phase 2: CMA-ES refinement")

                def _cma_callback(study, trial):
                    pbar2.update(1)
                    pbar2.set_postfix({"best_loss": f"{study.best_value:.4f}"})
            else:
                _cma_callback = None

            cma_study.optimize(
                self._objective_wrapper,
                n_trials=cma_trials,
                timeout=int(self.timeout * 0.2),
                callbacks=[_cma_callback] if _cma_callback else None,
            )

            if show_progress:
                pbar2.close()

            # Use whichever study found better params
            if cma_study.best_value < self.study.best_value:
                self.best_params = cma_study.best_params
                self._best_value = cma_study.best_value
            else:
                self.best_params = self.study.best_params
                self._best_value = self.study.best_value
        else:
            self.best_params = self.study.best_params
            self._best_value = self.study.best_value

        return self.best_params

    def _objective_wrapper(self, trial):
        """Optuna objective function wrapper."""
        params = self._define_search_space(trial)
        return self._compute_objective(params)

    def plot_results(self):
        """
        Show optimisation results: history, parameter importance, and
        a visual comparison of best synthetic vs real image.
        """
        from matplotlib import pyplot as plt

        if self.study is None:
            raise RuntimeError("No optimisation results. Call optimise() first.")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Optimisation history
        ax = axes[0]
        trials = [t for t in self.study.trials if t.value is not None and t.value != float("inf")]
        values = [t.value for t in trials]
        best_so_far = np.minimum.accumulate(values) if values else []
        ax.plot(values, alpha=0.3, label="Trial loss")
        ax.plot(best_so_far, color="red", linewidth=2, label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Composite Loss")
        ax.set_title("Optimisation History")
        ax.legend()

        # 2. Parameter importance
        ax = axes[1]
        try:
            importances = optuna.importance.get_param_importances(self.study)
            names = list(importances.keys())[:10]
            vals = [importances[n] for n in names]
            ax.barh(names, vals)
            ax.set_xlabel("Importance")
            ax.set_title("Parameter Importance")
        except Exception:
            ax.text(0.5, 0.5, "Not enough trials\nfor importance",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Parameter Importance")

        # 3. Best synthetic vs real comparison
        ax = axes[2]
        if self.best_params is not None:
            render_params = {k: v for k, v in self.best_params.items()
                            if k not in ("NA", "wavelength", "condenser")}
            # Apply PSF params if they were optimised
            if self.optimise_psf:
                psf_params = {f"_psf_{k}": v for k, v in self.best_params.items()
                              if k in ("NA", "wavelength", "condenser")}
                render_params.update(psf_params)
                self._apply_psf_params(render_params)

            try:
                synth, _ = self.renderer.render_synthetic(**render_params)
                # Side-by-side
                combined = np.hstack([
                    self.real_images[0][:synth.shape[0], :synth.shape[1]],
                    synth
                ])
                ax.imshow(combined, cmap="Greys_r")
                mid = synth.shape[1]
                ax.axvline(mid, color="red", linewidth=1, linestyle="--")
                ax.text(mid * 0.5, 5, "Real", ha="center", color="white",
                        fontsize=12, fontweight="bold")
                ax.text(mid * 1.5, 5, "Synthetic", ha="center", color="white",
                        fontsize=12, fontweight="bold")
            except Exception as e:
                ax.text(0.5, 0.5, f"Render error:\n{e}",
                        ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(f"Best Result (loss={self._best_value:.4f})")

        fig.tight_layout()
        plt.show()

    def apply_to_renderer(self):
        """
        Apply the best found parameters to the Renderer so that
        generate_training_data can be called directly.

        This creates a mock ipywidgets params object with the best
        parameters stored in kwargs, which is what generate_training_data
        expects.
        """
        if self.best_params is None:
            raise RuntimeError("No best parameters. Call optimise() first.")

        # Apply PSF params if optimised
        if self.optimise_psf:
            psf = self.renderer.PSF
            new_NA = self.best_params.get("NA", psf.NA)
            new_wl = self.best_params.get("wavelength", psf.wavelength)
            new_cond = self.best_params.get("condenser", psf.condenser)
            self.renderer.PSF = PSF_generator(
                radius=psf.radius,
                wavelength=new_wl,
                NA=new_NA,
                n=psf.n,
                resize_amount=self.renderer.simulation.resize_amount,
                pix_mic_conv=self.renderer.simulation.pix_mic_conv,
                apo_sigma=self.best_params.get("sigma", psf.apo_sigma),
                mode=psf.mode,
                condenser=new_cond,
            )
            self.renderer.PSF.calculate_PSF()

        # Set up image_params (needed by generate_test_comparison)
        stats = self._real_region_stats
        real = self.real_images[0]
        region_masks = self.renderer.auto_segment_regions(image=real, cells=self.cells)
        media_var = float(real[region_masks["media"]].var()) if real[region_masks["media"]].size > 0 else 0.0
        cell_var = float(real[region_masks["cell"]].var()) if real[region_masks["cell"]].size > 0 else 0.0
        device_var = float(real[region_masks["device"]].var()) if real[region_masks["device"]].size > 0 else 0.0

        self.renderer.image_params = (
            stats["media"], stats["cell"], stats["device"],
            np.array([stats["media"], stats["cell"], stats["device"]]),
            media_var, cell_var, device_var,
            np.array([media_var, cell_var, device_var]),
        )

        # Create a simple namespace that mimics ipywidgets interactive kwargs
        class _ParamsProxy:
            def __init__(self, kwargs):
                self.kwargs = kwargs

        render_keys = [
            "media_multiplier", "cell_multiplier", "device_multiplier",
            "sigma", "noise_var", "defocus", "match_fourier",
            "match_histogram", "match_noise", "halo_top_intensity",
            "halo_bottom_intensity", "halo_start", "halo_end",
        ]
        kwargs = {k: self.best_params[k] for k in render_keys if k in self.best_params}
        self.renderer.params = _ParamsProxy(kwargs)

    def get_best_params_summary(self):
        """
        Return a formatted summary of the best parameters found.

        Returns
        -------
        str
            Human-readable summary.
        """
        if self.best_params is None:
            return "No optimisation results yet."

        lines = [f"Best composite loss: {self._best_value:.6f}", ""]
        for k, v in sorted(self.best_params.items()):
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)
