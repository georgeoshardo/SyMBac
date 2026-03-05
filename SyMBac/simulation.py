import numpy as np
from joblib import Parallel, delayed
import pickle
import tempfile
from SyMBac.drawing import draw_scene, get_space_size, gen_cell_props_for_draw, generate_curve_props
from SyMBac.trench_geometry import  get_trench_segments
from SyMBac.config_models import SimulationSpec
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
    Segment-chain simulation entrypoint.

    Construct with a strict :class:`SyMBac.config_models.SimulationSpec` and
    run with :meth:`run`.

    Example:

    >>> from SyMBac.config_models import (
    ...     BrownianJitterSpec,
    ...     SimulationCellSpec,
    ...     SimulationGeometrySpec,
    ...     SimulationPhysicsSpec,
    ...     SimulationRuntimeSpec,
    ...     SimulationSpec,
    ... )
    >>> from SyMBac.simulation import Simulation
    >>> spec = SimulationSpec(
    ...     geometry=SimulationGeometrySpec(
    ...         trench_length=15,
    ...         trench_width=1.3,
    ...         pix_mic_conv=0.065,
    ...         resize_amount=3,
    ...     ),
    ...     cell=SimulationCellSpec(
    ...         cell_max_length=6.65,
    ...         cell_width=1.0,
    ...         max_length_std=0.0,
    ...         width_std=0.0,
    ...         lysis_p=0.0,
    ...     ),
    ...     physics=SimulationPhysicsSpec(gravity=0.0, phys_iters=15),
    ...     runtime=SimulationRuntimeSpec(sim_length=100, substeps=100, save_dir="/tmp/"),
    ...     brownian=BrownianJitterSpec(),
    ... )
    >>> my_simulation = Simulation(spec)
    >>> my_simulation.run(show_window=False)
    >>> my_simulation.draw_opl(do_transformation=True, label_masks=True)
    >>> my_simulation.visualise_in_napari()

    """

    def __init__(self, spec: SimulationSpec):
        if not isinstance(spec, SimulationSpec):
            raise TypeError("Simulation expects a SimulationSpec instance.")

        self.spec = spec
        self.trench_length = spec.geometry.trench_length
        self.trench_width = spec.geometry.trench_width
        self.cell_max_length = spec.cell.cell_max_length
        self.max_length_std = spec.cell.max_length_std
        self.cell_width = spec.cell.cell_width
        self.width_std = spec.cell.width_std
        self.lysis_p = spec.cell.lysis_p
        self.sim_length = spec.runtime.sim_length
        self.pix_mic_conv = spec.geometry.pix_mic_conv
        self.gravity = spec.physics.gravity
        self.phys_iters = spec.physics.phys_iters
        self.resize_amount = spec.geometry.resize_amount
        self.save_dir = spec.runtime.save_dir
        self.offset = 30
        self.load_sim_dir = spec.runtime.load_sim_dir
        self.substeps = spec.runtime.substeps
        self.width_upper_limit = spec.cell.width_upper_limit
        self.cell_config_template = spec.low_level.cell_config
        self.physics_config_template = spec.low_level.physics_config
        self.cell_config_overrides = dict(spec.low_level.cell_config_overrides)
        self.physics_config_overrides = dict(spec.low_level.physics_config_overrides)
        self.brownian_longitudinal_std = float(spec.brownian.longitudinal_std)
        self.brownian_transverse_std = float(spec.brownian.transverse_std)
        self.brownian_rotation_std = float(spec.brownian.rotation_std)
        self.brownian_persistence = float(spec.brownian.persistence)
        self.brownian_max_dx_fraction_of_trench_width = float(spec.brownian.max_dx_fraction_of_trench_width)
        self.brownian_max_dy_fraction_of_segment_radius = float(spec.brownian.max_dy_fraction_of_segment_radius)
        self.brownian_max_dy_px_floor = float(spec.brownian.max_dy_px_floor)
        self.brownian_max_dtheta = float(spec.brownian.max_dtheta)
        self.brownian_backoff_attempts = int(spec.brownian.backoff_attempts)
        self.brownian_application_mode = spec.brownian.application_mode
        self.brownian_projection_angular_damping = float(spec.brownian.projection_angular_damping)

        os.makedirs(self.save_dir, exist_ok=True)

        if self.load_sim_dir:
            print(
                "Loading previous simulation; no need to call run(). "
                "You still need to call draw_opl() with matching scale."
            )
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
            with open(os.path.join(self.load_sim_dir, "cell_timeseries.p"), "rb") as f:
                self.cell_timeseries = pickle.load(f)
            with open(os.path.join(self.load_sim_dir, "space_timeseries.p"), "rb") as f:
                self.space = pickle.load(f)

    


    def run(self, show_window=True):
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
        else:
            raise TypeError(
                "low_level.cell_config must be a dict or None."
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
        else:
            raise TypeError(
                "low_level.physics_config must be a dict or None."
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

    def draw_opl(self, do_transformation=True, label_masks=True, return_output=False):
        """
        Draw the optical path length images from the simulation.

        Uses segment-based drawing if the simulation was run with the new physics
        engine, otherwise falls back to the old vertex-based pipeline.

        :param bool do_transformation: Whether to bend cells (old pipeline only; new engine bends physically).
        :param bool label_masks: If True, masks are labelled per-cell; if False, binary.
        :param bool return_output: If True, return (OPL_scenes, masks).
        """
        # Detect new-engine snapshots (CellSnapshot with segment data)
        _has_segments = (
            hasattr(self, 'cell_timeseries')
            and len(self.cell_timeseries) > 0
            and len(self.cell_timeseries[0]) > 0
            and hasattr(self.cell_timeseries[0][0], 'segment_positions')
        )

        if _has_segments:
            from SyMBac.drawing import draw_scene_from_segments, get_space_size_from_segments

            # Trench geometry for the Renderer (same pymunk space, just parse it)
            self.main_segments = get_trench_segments(self.space)

            # Build per-frame segment data for drawing
            self.cell_timeseries_segments = []
            for frame_snapshots in self.cell_timeseries:
                frame_segments = [snap.to_segment_dict() for snap in frame_snapshots]
                self.cell_timeseries_segments.append(frame_segments)

            space_size = get_space_size_from_segments(self.cell_timeseries_segments, self.offset)
            self._space_size = space_size
            self._label_masks = label_masks
            self._do_transformation = do_transformation

            scenes = Parallel(n_jobs=-1)(delayed(draw_scene_from_segments)(
                frame_segments, space_size, self.offset, label_masks
            ) for frame_segments in tqdm(
                self.cell_timeseries_segments, desc='Rendering cell optical path lengths'))

            self.OPL_scenes = [s[0] for s in scenes]
            self.masks = [s[1] for s in scenes]

            # Build old-format properties for backwards compatibility
            self.cell_timeseries_properties = []
            for frame_snapshots in self.cell_timeseries:
                frame_props = []
                for snap in frame_snapshots:
                    angle_deg = np.rad2deg(snap.angle) + 90
                    pos = np.array([snap.position[0], snap.position[1]])
                    frame_props.append([
                        snap.length, snap.width, angle_deg, pos,
                        1.0, 1.0, 0.0, 20,
                        snap.pinching_sep, snap.mask_label, snap.ID
                    ])
                self.cell_timeseries_properties.append(frame_props)
        else:
            # Old pipeline (loaded from pickle or old Cell objects)
            self.main_segments = get_trench_segments(self.space)
            ID_props = generate_curve_props(self.cell_timeseries)

            self.cell_timeseries_properties = Parallel(n_jobs=-1)(
                delayed(gen_cell_props_for_draw)(a, ID_props) for a in tqdm(
                    self.cell_timeseries, desc='Extracting cell properties from the simulation'))

            space_size = get_space_size(self.cell_timeseries_properties)
            self._space_size = space_size
            self._do_transformation = do_transformation
            self._label_masks = label_masks

            scenes = Parallel(n_jobs=-1)(delayed(draw_scene)(
                cell_properties, do_transformation, space_size, self.offset, label_masks
            ) for cell_properties in tqdm(
                self.cell_timeseries_properties, desc='Rendering cell optical path lengths'))
            self.OPL_scenes = [_[0] for _ in scenes]
            self.masks = [_[1] for _ in scenes]

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
