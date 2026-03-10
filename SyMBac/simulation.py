import numpy as np
from joblib import Parallel, delayed
import pickle
import tempfile
from dataclasses import fields, is_dataclass
from SyMBac._deprecation import _UNSET, _require_provided
from SyMBac.drawing import draw_scene_from_segments, get_space_size_from_segments
from SyMBac.trench_geometry import  get_trench_segments
import napari
import os
from tqdm.auto import tqdm
from scipy.stats import norm

def _atomic_pickle_dump(payload, output_path):
    output_dir = os.path.dirname(output_path) or "."
    fd, temp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".p", dir=output_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


class Simulation:
    
    """
    Class for instantiating Simulation objects. These are the basic objects used to run all SyMBac simulations. This
    class is used to parameterise simulations, run them, draw optical path length images, and then visualise them.

    Example:

    >>> from SyMBac.simulation import Simulation
    >>> my_simulation = Simulation(
            trench_length=15,
            trench_width=1.3,
            cell_max_length=6.65, #6, long cells # 1.65 short cells
            cell_width= 1, #1 long cells # 0.95 short cells
            sim_length = 100,
            pix_mic_conv = 0.065,
            gravity=0,
            phys_iters=15,
            max_length_std = 0.,
            width_std = 0.,
            lysis_p = 0.,
            save_dir="/tmp/",
            resize_amount = 3,
            substeps = 100
        )
    >>> my_simulation.run_simulation(show_window=False)
    >>> my_simulation.draw_simulation_OPL(label_masks=True)
    >>> my_simulation.visualise_in_napari()

    """

    def __init__(
        self,
        trench_length,
        trench_width,
        cell_max_length,
        *,
        max_length_std=_UNSET,
        cell_width=_UNSET,
        width_std=_UNSET,
        lysis_p=_UNSET,
        sim_length=_UNSET,
        pix_mic_conv=_UNSET,
        gravity=_UNSET,
        phys_iters=_UNSET,
        resize_amount=_UNSET,
        save_dir=_UNSET,
        substeps=100,
        width_upper_limit=None,
        load_sim_dir=None,
        cell_config=None,
        physics_config=None,
        cell_config_overrides=None,
        physics_config_overrides=None,
        brownian_longitudinal_std=0.0,
        brownian_transverse_std=0.0,
        brownian_rotation_std=0.0,
        brownian_persistence=0.85,
        brownian_max_dx_fraction_of_trench_width=0.20,
        brownian_max_dy_fraction_of_segment_radius=0.75,
        brownian_max_dy_px_floor=1.0,
        brownian_max_dtheta=0.03,
        brownian_backoff_attempts=5,
        brownian_application_mode="teleport",
        brownian_projection_angular_damping=0.35,
    ):
        """
        Initialising a Simulation object

        Parameters
        ----------
        trench_length : float
            Length of a mother machine trench (micron)
        trench_width : float
            Width of a mother machine trench (micron)
        cell_max_length : float
            Maximum length a cell can reach before dividing (micron)
        cell_width : float
            the average cell width in the simulation (micron)
        pix_mic_conv : float
            The micron/pixel size of the image
        gravity : float
            Pressure forcing cells into the trench. Typically left at zero, but can be varied if cells start to fall into
            each other or if the simulation behaves strangely.
        phys_iters : int
            Number of physics iterations per simulation frame. Increase to resolve collisions if cells are falling into one
            another, but decrease if cells begin to repel one another too much (too high a value causes cells to bounce off
            each other very hard). 20 is a good starting point
        max_length_std : float
            Standard deviation of the per-cell division length distribution.
            New maximum lengths are sampled from a normal distribution
            centered on ``cell_max_length`` at birth/division.
        width_std : float
            Standard deviation of the per-cell width distribution.
            New width targets are sampled from a normal distribution centered
            on ``cell_width`` at division, then relaxed smoothly over time.
        save_dir : str
            Location to save simulation output
        lysis_p : float
            probability of cell lysis
        sim_length : int
            number of frames to simulate (where each one is dt). Start with 200-1000, and grow your simulation from there.
        resize_amount : int
            This is the "upscaling" factor for the simulation and the entire image generation process. Must be kept constant
            across the image generation pipeline. Starting value of 3 recommended.
        substeps : int, optional
            Number of physics sub-steps to execute per output frame.
            Defaults to ``100`` when omitted for backward compatibility.
        width_upper_limit : float or None, optional
            Hard upper limit on cell width in microns. Useful for ensuring
            cells cannot grow wider than the mother machine trench. Set to
            slightly below ``trench_width`` to prevent jamming. ``None``
            (default) means no limit.
        load_sim_dir : str
            The directory if you wish to load a previously completed simulation
        cell_config : SyMBac.physics.config.CellConfig or dict or None, optional
            Optional full low-level cell physics configuration template.
            If provided, this is used as the base cell configuration instead
            of internally computed defaults.
        physics_config : SyMBac.physics.config.PhysicsConfig or dict or None, optional
            Optional full low-level global physics configuration template.
            If provided, this is used as the base physics configuration instead
            of internally computed defaults.
        cell_config_overrides : dict or None, optional
            Key-value overrides applied to the resolved cell configuration
            template (whether default or user-provided).
        physics_config_overrides : dict or None, optional
            Key-value overrides applied to the resolved physics configuration
            template (whether default or user-provided).
        brownian_longitudinal_std : float, optional
            Standard deviation of per-output-frame cell translation along the
            trench axis (microns). Applied as temporally correlated (OU-like)
            rigid-body jitter.
        brownian_transverse_std : float, optional
            Standard deviation of per-output-frame cell translation across the
            trench axis (microns). Applied as temporally correlated (OU-like)
            rigid-body jitter.
        brownian_rotation_std : float, optional
            Standard deviation of per-output-frame rigid-body rotation
            perturbation (radians).
        brownian_persistence : float, optional
            Temporal persistence for Brownian jitter in [0, 1). Values near 0
            are white-noise-like; values near 1 produce slowly drifting motion.
        brownian_max_dx_fraction_of_trench_width : float, optional
            Hard cap for per-frame transverse translation. ``|dx|`` is clipped
            to this fraction of trench width in simulation pixels.
        brownian_max_dy_fraction_of_segment_radius : float, optional
            Hard cap component for per-frame longitudinal translation based on
            current segment radius.
        brownian_max_dy_px_floor : float, optional
            Lower bound for the longitudinal cap in simulation pixels.
        brownian_max_dtheta : float, optional
            Hard cap for per-frame rotational jitter (radians).
        brownian_backoff_attempts : int, optional
            Number of geometric backoff retries before rolling back jitter for
            a frame.
        brownian_application_mode : {"teleport", "velocity", "impulse"}, optional
            Brownian application strategy. ``teleport`` applies a rigid-body
            pose perturbation directly after sub-stepping. ``velocity`` and
            ``impulse`` convert the sampled perturbation into dynamic
            perturbations before sub-stepping so collisions are resolved by
            Pymunk.
        brownian_projection_angular_damping : float, optional
            Angular damping factor applied when projection safety correction is
            needed in ``velocity`` or ``impulse`` modes. Must be in ``[0, 1]``.
        """
        api_name = f"{self.__class__.__name__}.__init__()"
        _require_provided(api_name, "cell_width", cell_width)
        _require_provided(api_name, "lysis_p", lysis_p)
        _require_provided(api_name, "sim_length", sim_length)
        _require_provided(api_name, "pix_mic_conv", pix_mic_conv)
        _require_provided(api_name, "gravity", gravity)
        _require_provided(api_name, "phys_iters", phys_iters)
        _require_provided(api_name, "resize_amount", resize_amount)
        _require_provided(api_name, "save_dir", save_dir)

        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps <= 0:
            raise ValueError("substeps must be an integer greater than 0.")
        if cell_config_overrides is not None and not isinstance(cell_config_overrides, dict):
            raise TypeError("cell_config_overrides must be a dict when provided.")
        if physics_config_overrides is not None and not isinstance(physics_config_overrides, dict):
            raise TypeError("physics_config_overrides must be a dict when provided.")
        for name, value in (
            ("brownian_longitudinal_std", brownian_longitudinal_std),
            ("brownian_transverse_std", brownian_transverse_std),
            ("brownian_rotation_std", brownian_rotation_std),
        ):
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative float.")
        if isinstance(brownian_persistence, bool) or not (0.0 <= brownian_persistence < 1.0):
            raise ValueError("brownian_persistence must be in [0.0, 1.0).")
        for name, value in (
            ("brownian_max_dx_fraction_of_trench_width", brownian_max_dx_fraction_of_trench_width),
            ("brownian_max_dy_fraction_of_segment_radius", brownian_max_dy_fraction_of_segment_radius),
            ("brownian_max_dy_px_floor", brownian_max_dy_px_floor),
            ("brownian_max_dtheta", brownian_max_dtheta),
        ):
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be > 0.")
        if (
            isinstance(brownian_backoff_attempts, bool)
            or not isinstance(brownian_backoff_attempts, int)
            or brownian_backoff_attempts < 1
        ):
            raise ValueError("brownian_backoff_attempts must be an integer >= 1.")
        if brownian_application_mode not in {"teleport", "velocity", "impulse"}:
            raise ValueError(
                "brownian_application_mode must be one of: 'teleport', 'velocity', 'impulse'."
            )
        if (
            isinstance(brownian_projection_angular_damping, bool)
            or not (0.0 <= brownian_projection_angular_damping <= 1.0)
        ):
            raise ValueError("brownian_projection_angular_damping must be in [0.0, 1.0].")

        self.trench_length = trench_length
        self.trench_width = trench_width
        self.cell_max_length = cell_max_length
        self.max_length_std = max_length_std
        self.cell_width = cell_width
        self.width_std = width_std
        self.lysis_p = lysis_p
        self.sim_length = sim_length
        self.pix_mic_conv = pix_mic_conv
        self.gravity = gravity
        self.phys_iters = phys_iters
        self.resize_amount = resize_amount
        self.save_dir = save_dir
        self.offset = 30
        self.load_sim_dir = load_sim_dir
        self.substeps = substeps
        self.width_upper_limit = width_upper_limit
        self.cell_config_template = cell_config
        self.physics_config_template = physics_config
        self.cell_config_overrides = dict(cell_config_overrides or {})
        self.physics_config_overrides = dict(physics_config_overrides or {})
        self.brownian_longitudinal_std = float(brownian_longitudinal_std)
        self.brownian_transverse_std = float(brownian_transverse_std)
        self.brownian_rotation_std = float(brownian_rotation_std)
        self.brownian_persistence = float(brownian_persistence)
        self.brownian_max_dx_fraction_of_trench_width = float(brownian_max_dx_fraction_of_trench_width)
        self.brownian_max_dy_fraction_of_segment_radius = float(brownian_max_dy_fraction_of_segment_radius)
        self.brownian_max_dy_px_floor = float(brownian_max_dy_px_floor)
        self.brownian_max_dtheta = float(brownian_max_dtheta)
        self.brownian_backoff_attempts = int(brownian_backoff_attempts)
        self.brownian_application_mode = brownian_application_mode
        self.brownian_projection_angular_damping = float(brownian_projection_angular_damping)

        os.makedirs(self.save_dir, exist_ok=True)

        if self.load_sim_dir:
            print("Loading previous simulation, no need to call run_simulation method, but you still need to run OPL drawing and correctly define the scale")
            required_artifacts = ("cell_timeseries.p", "space_timeseries.p")
            missing_artifacts = [
                name
                for name in required_artifacts
                if not os.path.exists(os.path.join(self.load_sim_dir, name))
            ]
            if missing_artifacts:
                raise FileNotFoundError(
                    (
                        f"Could not load simulation from '{self.load_sim_dir}'. "
                        "Expected files: cell_timeseries.p, space_timeseries.p. "
                        f"Missing: {', '.join(missing_artifacts)}."
                    )
                )
            with open(os.path.join(self.load_sim_dir, "cell_timeseries.p"), 'rb') as f:
                self.cell_timeseries = pickle.load(f)
            with open(os.path.join(self.load_sim_dir, "space_timeseries.p"), 'rb') as f:
                self.space = pickle.load(f)
            if not self._loaded_timeseries_uses_segments(self.cell_timeseries):
                raise ValueError(
                    "Legacy simulation artifacts are no longer supported. "
                    "Regenerate the simulation with the current segment-based engine."
                )

    @staticmethod
    def _loaded_timeseries_uses_segments(cell_timeseries):
        if not isinstance(cell_timeseries, (list, tuple)):
            return False
        for frame in cell_timeseries:
            if not isinstance(frame, (list, tuple)):
                return False
            for cell in frame:
                if not hasattr(cell, "segment_positions"):
                    return False
        return True



    def run_simulation(self, show_window=True):
        """
        Run the simulation using the segment-chain physics engine.

        :param bool show_window: If True, opens a live Pyglet window with physics debug drawing while simulating.
        """
        from SyMBac.physics.config import CellConfig, PhysicsConfig
        from SyMBac.physics.simulator import Simulator
        from SyMBac.physics import microfluidic_geometry
        from SyMBac.cell_snapshot import CellSnapshot

        # --- Scale parameters from microns to pixels ---
        scale_factor = (1 / self.pix_mic_conv) * self.resize_amount
        radius_scale = self.cell_width * scale_factor / 2 / 10  # ratio vs SyMBac_2's SEGMENT_RADIUS=10

        SEGMENT_RADIUS = self.cell_width * scale_factor / 2
        GRANULARITY = 4
        BASE_MAX_LENGTH = self.cell_max_length * scale_factor
        JOINT_DISTANCE = SEGMENT_RADIUS / GRANULARITY
        SEED_CELL_SEGMENTS = max(3, int(self.cell_max_length * 0.5 * scale_factor / JOINT_DISTANCE))

        MAX_LENGTH_STD = max(0.0, self.max_length_std * scale_factor)
        WIDTH_STD = max(0.0, self.width_std * scale_factor)
        WIDTH_UPPER_LIMIT = (self.width_upper_limit * scale_factor / 2) if self.width_upper_limit is not None else None

        GROWTH_RATE = 0.5 * SEGMENT_RADIUS

        trench_length_px = self.trench_length * scale_factor
        trench_width_px = self.trench_width * scale_factor
        SUB_STEPS = self.substeps
        brownian_longitudinal_std_px = self.brownian_longitudinal_std * scale_factor
        brownian_transverse_std_px = self.brownian_transverse_std * scale_factor
        brownian_state = {}  # group_id -> np.array([dy, dx, dtheta])

        default_cell_kwargs = dict(
            GRANULARITY=GRANULARITY,
            SEGMENT_RADIUS=SEGMENT_RADIUS,
            SEGMENT_MASS=1.0,
            GROWTH_RATE=GROWTH_RATE,
            MIN_LENGTH_AFTER_DIVISION=max(3, GRANULARITY),
            MAX_LENGTH_STD=MAX_LENGTH_STD,
            BASE_MAX_LENGTH=BASE_MAX_LENGTH,
            WIDTH_STD=WIDTH_STD,
            SEED_CELL_SEGMENTS=SEED_CELL_SEGMENTS,
            PIVOT_JOINT_STIFFNESS=5_000 * radius_scale,
            NOISE_STRENGTH=0.05,
            START_POS=(35, trench_width_px / 2 + SEGMENT_RADIUS * 3),
            START_ANGLE=np.pi / 2,
            SEPTUM_DURATION=1.5,
            ROTARY_LIMIT_JOINT=True,
            MAX_BEND_ANGLE=0.005,
            STIFFNESS=300_000 * radius_scale,
            SIMPLE_LENGTH=False,
            WIDTH_UPPER_LIMIT=WIDTH_UPPER_LIMIT,
        )
        if self.cell_config_template is None:
            resolved_cell_kwargs = dict(default_cell_kwargs)
        elif isinstance(self.cell_config_template, dict):
            resolved_cell_kwargs = dict(self.cell_config_template)
        elif is_dataclass(self.cell_config_template):
            resolved_cell_kwargs = {
                field.name: getattr(self.cell_config_template, field.name)
                for field in fields(self.cell_config_template)
                if field.init
            }
        else:
            raise TypeError(
                "cell_config must be a CellConfig-like dataclass instance, a dict, or None."
            )
        resolved_cell_kwargs.update(self.cell_config_overrides)
        cell_config = CellConfig(**resolved_cell_kwargs)

        default_physics_kwargs = dict(
            ITERATIONS=self.phys_iters * 8,
            DAMPING=0.5,
            GRAVITY=(0, self.gravity),
        )
        if self.physics_config_template is None:
            resolved_physics_kwargs = dict(default_physics_kwargs)
        elif isinstance(self.physics_config_template, dict):
            resolved_physics_kwargs = dict(self.physics_config_template)
        elif is_dataclass(self.physics_config_template):
            resolved_physics_kwargs = {
                field.name: getattr(self.physics_config_template, field.name)
                for field in fields(self.physics_config_template)
                if field.init
            }
        else:
            raise TypeError(
                "physics_config must be a PhysicsConfig-like dataclass instance, a dict, or None."
            )
        resolved_physics_kwargs.update(self.physics_config_overrides)
        physics_config = PhysicsConfig(**resolved_physics_kwargs)
        self._resolved_cell_config = cell_config
        self._resolved_physics_config = physics_config
        brownian_mode = self.brownian_application_mode
        dynamic_brownian_mode = brownian_mode in {"velocity", "impulse"}
        frame_dt = max(float(physics_config.DT) * float(SUB_STEPS), 1e-12)

        # --- Lineage tracking ---
        lineage_info = {}  # group_id -> {mother_mask_label, generation}
        just_divided_this_frame = set()

        def create_trench(sim):
            microfluidic_geometry.trench_creator(
                width=trench_width_px,
                trench_length=trench_length_px,
                global_xy=(35, 0),
                space=sim.space,
            )

        def remove_out_of_bounds(sim):
            trench_open_end = trench_width_px / 2 + trench_length_px
            for cell in sim.colony.cells[:]:
                segments = getattr(cell.physics_representation, "segments", None)
                if not segments:
                    continue
                segment_radii = [float(getattr(segment, "radius", SEGMENT_RADIUS)) for segment in segments]
                segment_ys = [float(segment.position[1]) for segment in segments]
                min_radius = min(segment_radii) if segment_radii else SEGMENT_RADIUS
                max_radius = max(segment_radii) if segment_radii else SEGMENT_RADIUS
                below_floor = min(segment_ys) < -0.25 * min_radius
                above_open_end = max(segment_ys) > trench_open_end + 0.25 * max_radius
                if below_floor or above_open_end:
                    sim.colony.delete_cell(cell)

        def handle_lysis(sim):
            if self.lysis_p <= 0:
                return
            for cell in sim.colony.cells[:]:
                if len(sim.colony.cells) <= 1:
                    break
                if norm.rvs() <= norm.ppf(self.lysis_p):
                    sim.colony.delete_cell(cell)

        def apply_rigid_body_brownian_jitter(sim):
            if (
                brownian_longitudinal_std_px <= 0
                and brownian_transverse_std_px <= 0
                and self.brownian_rotation_std <= 0
            ):
                return

            rho = self.brownian_persistence
            noise_scale = np.sqrt(max(0.0, 1.0 - rho ** 2))
            live_ids = set()
            trench_center_x = 35.0
            trench_half_width = trench_width_px / 2.0
            trench_y_min = 0.0
            trench_y_max = trench_width_px / 2.0 + trench_length_px
            # Hard caps against unrealistically large rigid-body jumps.
            max_dx_abs = self.brownian_max_dx_fraction_of_trench_width * trench_width_px
            max_dy_abs = max(
                self.brownian_max_dy_px_floor,
                self.brownian_max_dy_fraction_of_segment_radius * SEGMENT_RADIUS,
            )
            max_dtheta_abs = self.brownian_max_dtheta
            max_backoff_attempts = self.brownian_backoff_attempts
            enforce_open_end_cap = not dynamic_brownian_mode

            def _positions_inside_trench(segments, positions):
                for segment, position in zip(segments, positions):
                    radius = float(getattr(segment, "radius", SEGMENT_RADIUS))
                    x = float(position[0])
                    y = float(position[1])
                    if x < (trench_center_x - trench_half_width + radius):
                        return False
                    if x > (trench_center_x + trench_half_width - radius):
                        return False
                    if y < (trench_y_min + 0.25 * radius):
                        return False
                    if enforce_open_end_cap and y > (trench_y_max - 0.25 * radius):
                        return False
                return True

            def _build_trial_positions(old_positions, center_vec, dx_value, dy_value, dtheta_value):
                vec_type = center_vec.__class__
                translation = vec_type(dx_value, dy_value)
                return [
                    center_vec + (old_pos - center_vec).rotated(dtheta_value) + translation
                    for old_pos in old_positions
                ]

            def _apply_teleport(segment_bodies, old_positions, old_angles, center_vec, dx_value, dy_value, dtheta_value):
                trial_positions = _build_trial_positions(
                    old_positions=old_positions,
                    center_vec=center_vec,
                    dx_value=dx_value,
                    dy_value=dy_value,
                    dtheta_value=dtheta_value,
                )
                for body, trial_pos, old_angle in zip(segment_bodies, trial_positions, old_angles):
                    body.position = trial_pos
                    body.angle = old_angle + dtheta_value

            def _apply_dynamic_jitter(segment_bodies, center, dx_value, dy_value, dtheta_value):
                translational_vx = dx_value / frame_dt
                translational_vy = dy_value / frame_dt
                angular_velocity = dtheta_value / frame_dt

                for body in segment_bodies:
                    vec_type = body.position.__class__
                    rel_x = float(body.position[0] - center[0])
                    rel_y = float(body.position[1] - center[1])
                    jitter_velocity = vec_type(
                        translational_vx - angular_velocity * rel_y,
                        translational_vy + angular_velocity * rel_x,
                    )

                    current_velocity = getattr(body, "velocity", vec_type(0.0, 0.0))
                    current_velocity = vec_type(float(current_velocity[0]), float(current_velocity[1]))
                    current_angular_velocity = float(getattr(body, "angular_velocity", 0.0))

                    if brownian_mode == "velocity":
                        body.velocity = current_velocity + jitter_velocity
                    else:
                        if hasattr(body, "apply_impulse_at_local_point"):
                            body_mass = float(getattr(body, "mass", 1.0))
                            body.apply_impulse_at_local_point(jitter_velocity * body_mass)
                        else:
                            body.velocity = current_velocity + jitter_velocity

                    body.angular_velocity = current_angular_velocity + angular_velocity

            for cell in sim.colony.cells:
                group_id = getattr(cell, "group_id", None)
                if group_id is None:
                    continue
                live_ids.add(group_id)

                physics_rep = getattr(cell, "physics_representation", None)
                segments = getattr(physics_rep, "segments", None)
                if not segments:
                    continue

                prev_state = brownian_state.get(group_id, np.zeros(3, dtype=float))
                dy = rho * prev_state[0] + noise_scale * np.random.normal(0.0, brownian_longitudinal_std_px)
                dx = rho * prev_state[1] + noise_scale * np.random.normal(0.0, brownian_transverse_std_px)
                dtheta = rho * prev_state[2] + noise_scale * np.random.normal(0.0, self.brownian_rotation_std)
                dy = float(np.clip(dy, -max_dy_abs, max_dy_abs))
                dx = float(np.clip(dx, -max_dx_abs, max_dx_abs))
                dtheta = float(np.clip(dtheta, -max_dtheta_abs, max_dtheta_abs))

                segment_bodies = [getattr(segment, "body", None) for segment in segments]
                segment_bodies = [body for body in segment_bodies if body is not None]
                if not segment_bodies:
                    continue

                vec_type = segment_bodies[0].position.__class__
                center = np.mean(
                    np.array([[float(body.position[0]), float(body.position[1])] for body in segment_bodies], dtype=float),
                    axis=0,
                )
                center_vec = vec_type(center[0], center[1])
                old_positions = [vec_type(float(body.position[0]), float(body.position[1])) for body in segment_bodies]
                old_angles = [float(body.angle) for body in segment_bodies]

                accepted_state = None
                for attempt in range(max_backoff_attempts):
                    scale = 0.5 ** attempt
                    trial_dx = dx * scale
                    trial_dy = dy * scale
                    trial_dtheta = dtheta * scale
                    trial_positions = _build_trial_positions(
                        old_positions=old_positions,
                        center_vec=center_vec,
                        dx_value=trial_dx,
                        dy_value=trial_dy,
                        dtheta_value=trial_dtheta,
                    )

                    if _positions_inside_trench(segments, trial_positions):
                        accepted_state = np.array([trial_dy, trial_dx, trial_dtheta], dtype=float)
                        break

                if accepted_state is None:
                    brownian_state[group_id] = np.zeros(3, dtype=float)
                else:
                    if dynamic_brownian_mode:
                        _apply_dynamic_jitter(
                            segment_bodies=segment_bodies,
                            center=center,
                            dx_value=float(accepted_state[1]),
                            dy_value=float(accepted_state[0]),
                            dtheta_value=float(accepted_state[2]),
                        )
                    else:
                        _apply_teleport(
                            segment_bodies=segment_bodies,
                            old_positions=old_positions,
                            old_angles=old_angles,
                            center_vec=center_vec,
                            dx_value=float(accepted_state[1]),
                            dy_value=float(accepted_state[0]),
                            dtheta_value=float(accepted_state[2]),
                        )
                    brownian_state[group_id] = accepted_state

            stale_ids = [group_id for group_id in brownian_state.keys() if group_id not in live_ids]
            for group_id in stale_ids:
                brownian_state.pop(group_id, None)

        def project_segments_inside_trench(sim):
            if not dynamic_brownian_mode:
                return

            trench_center_x = 35.0
            trench_half_width = trench_width_px / 2.0
            trench_y_min = 0.0
            trench_y_max = trench_width_px / 2.0 + trench_length_px
            angular_damping = self.brownian_projection_angular_damping

            for cell in sim.colony.cells:
                group_id = getattr(cell, "group_id", None)
                physics_rep = getattr(cell, "physics_representation", None)
                segments = getattr(physics_rep, "segments", None)
                if not segments:
                    continue

                projected = False
                for segment in segments:
                    body = getattr(segment, "body", None)
                    if body is None:
                        continue

                    radius = float(getattr(segment, "radius", SEGMENT_RADIUS))
                    min_x = trench_center_x - trench_half_width + radius
                    max_x = trench_center_x + trench_half_width - radius
                    min_y = trench_y_min + 0.25 * radius

                    x = float(body.position[0])
                    y = float(body.position[1])
                    clamped_x = min(max(x, min_x), max_x)
                    clamped_y = max(y, min_y)

                    if clamped_x == x and clamped_y == y:
                        continue

                    vec_type = body.position.__class__
                    body.position = vec_type(clamped_x, clamped_y)
                    if hasattr(body, "velocity"):
                        velocity = getattr(body, "velocity")
                        velocity = vec_type(float(velocity[0]), float(velocity[1]))
                        if clamped_x != x:
                            velocity = vec_type(0.0, float(velocity[1]))
                        if clamped_y != y:
                            velocity = vec_type(float(velocity[0]), 0.0)
                        body.velocity = velocity
                    projected = True

                if projected:
                    for segment in segments:
                        body = getattr(segment, "body", None)
                        if body is None or not hasattr(body, "angular_velocity"):
                            continue
                        body.angular_velocity = float(body.angular_velocity) * angular_damping
                    if group_id in brownian_state:
                        brownian_state[group_id][2] *= angular_damping

        def track_lineage(mother, daughter):
            mother_gen = lineage_info.get(mother.group_id, {}).get('generation', 0)
            lineage_info[daughter.group_id] = {
                'mother_mask_label': mother.group_id,
                'generation': mother_gen + 1,
            }
            if mother.group_id not in lineage_info:
                lineage_info[mother.group_id] = {
                    'mother_mask_label': None,
                    'generation': 0,
                }

        def mark_just_divided(mother, daughter):
            just_divided_this_frame.add(daughter.group_id)
            just_divided_this_frame.add(mother.group_id)

        def resample_max_lengths_after_division(mother, daughter):
            # Keep each generation centered on BASE_MAX_LENGTH rather than inheriting drift.
            mother.max_length = mother.sample_max_length()
            daughter.max_length = daughter.sample_max_length()

        post_step_hooks = [remove_out_of_bounds]
        # Lysis is handled once per output frame (not per sub-step) to keep
        # the effective probability independent of the substeps setting.

        def cell_growth_rate_updater(cell):
            cell.update_width_transition(physics_config.DT)
            compression_ratio = cell.physics_representation.get_compression_ratio()
            cell.adjusted_growth_rate = cell.config.GROWTH_RATE * compression_ratio ** 4

        def post_growth_width_sync(cell):
            cell.apply_current_width_to_segments()

        sim = Simulator(
            physics_config=physics_config,
            initial_cell_config=cell_config,
            post_init_hooks=[create_trench],
            pre_cell_grow_hooks=[cell_growth_rate_updater],
            post_cell_grow_hooks=[post_growth_width_sync],
            post_step_hooks=post_step_hooks,
            post_division_hooks=[resample_max_lengths_after_division, track_lineage, mark_just_divided],
        )

        # Initialize lineage for seed cell(s)
        for cell in sim.colony.cells:
            lineage_info[cell.group_id] = {
                'mother_mask_label': None,
                'generation': 0,
            }

        # --- Run simulation loop ---
        cell_timeseries = []
        total_frames = self.sim_length + 2

        def step_and_capture_frame(frame_idx):
            just_divided_this_frame.clear()
            if dynamic_brownian_mode:
                apply_rigid_body_brownian_jitter(sim)
            for _ in range(SUB_STEPS):
                sim.step()
            # Lysis once per output frame so probability is independent of substeps
            if self.lysis_p > 0:
                handle_lysis(sim)
            if dynamic_brownian_mode:
                project_segments_inside_trench(sim)
            else:
                apply_rigid_body_brownian_jitter(sim)

            # Skip first 2 warmup frames (matching old engine)
            if frame_idx >= 2:
                frame_snapshots = []
                for cell in sim.colony.cells:
                    info = lineage_info.get(cell.group_id, {})
                    snap = CellSnapshot(
                        simcell=cell,
                        t=frame_idx - 2,
                        mother_mask_label=info.get('mother_mask_label'),
                        generation=info.get('generation', 0),
                        just_divided=cell.group_id in just_divided_this_frame,
                        lysis_p=self.lysis_p,
                    )
                    frame_snapshots.append(snap)
                cell_timeseries.append(frame_snapshots)

        if show_window:
            try:
                import pyglet
                from pymunk.pyglet_util import DrawOptions
            except ImportError as e:
                raise ImportError(
                    "show_window=True requires pyglet. Install it or rerun with show_window=False."
                ) from e

            import threading
            import time

            window = pyglet.window.Window(900, 900, "SyMBac", resizable=True)
            draw_options = DrawOptions()
            draw_options.shape_outline_color = (235, 235, 235, 255)
            progress_bar = tqdm(total=total_frames, desc='Running simulation')
            state = {
                "frame_idx": 0,
                "progress_idx": 0,
                "stopped": False,
                "done": False,
                "worker_error": None,
            }
            sim_lock = threading.Lock()
            stop_event = threading.Event()

            def simulation_worker():
                try:
                    for frame_idx in range(total_frames):
                        if stop_event.is_set():
                            break
                        with sim_lock:
                            step_and_capture_frame(frame_idx)
                        state["frame_idx"] = frame_idx + 1
                        time.sleep(0)
                except Exception as e:
                    state["worker_error"] = e
                finally:
                    state["done"] = True

            worker_thread = threading.Thread(target=simulation_worker, daemon=True)
            worker_thread.start()

            @window.event
            def on_draw():
                window.clear()
                with sim_lock:
                    sim.space.debug_draw(draw_options)

            def stop_loop():
                if state["stopped"]:
                    return
                state["stopped"] = True
                stop_event.set()
                pyglet.clock.unschedule(update_ui)
                if state["frame_idx"] > state["progress_idx"]:
                    progress_bar.update(state["frame_idx"] - state["progress_idx"])
                    state["progress_idx"] = state["frame_idx"]
                progress_bar.close()
                if worker_thread.is_alive():
                    worker_thread.join(timeout=1.0)
                if not window.has_exit:
                    window.close()
                pyglet.app.exit()

            @window.event
            def on_key_press(symbol, modifier):
                if symbol in (pyglet.window.key.E, pyglet.window.key.ESCAPE):
                    stop_loop()

            @window.event
            def on_close():
                stop_loop()

            def update_ui(_dt):
                if state["frame_idx"] > state["progress_idx"]:
                    progress_bar.update(state["frame_idx"] - state["progress_idx"])
                    state["progress_idx"] = state["frame_idx"]
                if state["done"]:
                    stop_loop()

            pyglet.clock.schedule_interval(update_ui, 1 / 60.0)
            pyglet.app.run()
            if state["worker_error"] is not None:
                raise RuntimeError("Simulation worker failed during show_window=True execution.") from state["worker_error"]
        else:
            for frame_idx in tqdm(range(total_frames), desc='Running simulation'):
                step_and_capture_frame(frame_idx)

        self.cell_timeseries = cell_timeseries
        self.space = sim.space
        self.historic_cells = []
        _atomic_pickle_dump(self.cell_timeseries, os.path.join(self.save_dir, "cell_timeseries.p"))
        _atomic_pickle_dump(self.space, os.path.join(self.save_dir, "space_timeseries.p"))

    def draw_simulation_OPL(self, label_masks=True, return_output=False):
        """
        Draw the optical path length images from the simulation.

        Uses the maintained segment-based drawing pipeline.

        :param bool label_masks: If True, masks are labelled per-cell; if False, binary.
        :param bool return_output: If True, return (OPL_scenes, masks).
        """
        if not self._loaded_timeseries_uses_segments(getattr(self, "cell_timeseries", [])):
            raise ValueError(
                "Simulation cell_timeseries must contain segment-based snapshots. "
                "Legacy rigid-body data is no longer supported."
            )

        self.main_segments = get_trench_segments(self.space)

        self.cell_timeseries_segments = []
        for frame_snapshots in self.cell_timeseries:
            frame_segments = [snap.to_segment_dict() for snap in frame_snapshots]
            self.cell_timeseries_segments.append(frame_segments)

        space_size = get_space_size_from_segments(self.cell_timeseries_segments, self.offset)
        self._space_size = space_size
        self._label_masks = label_masks

        scenes = Parallel(n_jobs=-1)(delayed(draw_scene_from_segments)(
            frame_segments, space_size, self.offset, label_masks
        ) for frame_segments in tqdm(
            self.cell_timeseries_segments, desc='Rendering cell optical path lengths'))

        self.OPL_scenes = [s[0] for s in scenes]
        self.masks = [s[1] for s in scenes]

        if return_output:
            return self.OPL_scenes, self.masks

    def visualise_in_napari(self):
        """
        Opens a napari window allowing you to visualise the simulation, with both masks, OPL images, interactively.
        :return:
        """
        
        viewer = napari.Viewer()
        viewer.add_image(np.array(self.OPL_scenes), name='OPL scenes')
        viewer.add_labels(np.array(self.masks), name='Synthetic masks')
        napari.run()
