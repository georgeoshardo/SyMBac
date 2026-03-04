import numpy as np
from joblib import Parallel, delayed
import pickle
import tempfile
from SyMBac._deprecation import _UNSET, _resolve_deprecated_parameter, _require_provided
from SyMBac.drawing import draw_scene, get_space_size, gen_cell_props_for_draw, generate_curve_props
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
    >>> my_simulation.draw_simulation_OPL(do_transformation=True, label_masks=True)
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
        load_sim_dir=None,
        max_length_var=_UNSET,
        width_var=_UNSET,
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
        load_sim_dir : str
            The directory if you wish to load a previously completed simulation
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

        max_length_std, _ = _resolve_deprecated_parameter(
            api_name=api_name,
            new_name="max_length_std",
            new_value=max_length_std,
            legacy_name="max_length_var",
            legacy_value=max_length_var,
            compatibility_note="`max_length_var` is interpreted as a standard deviation in this API.",
        )
        width_std, _ = _resolve_deprecated_parameter(
            api_name=api_name,
            new_name="width_std",
            new_value=width_std,
            legacy_name="width_var",
            legacy_value=width_var,
            compatibility_note="`width_var` is interpreted as a standard deviation in this API.",
        )

        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps <= 0:
            raise ValueError("substeps must be an integer greater than 0.")

        self.trench_length = trench_length
        self.trench_width = trench_width
        self.cell_max_length = cell_max_length
        self.max_length_std = max_length_std
        self.max_length_var = max_length_std
        self.cell_width = cell_width
        self.width_std = width_std
        self.width_var = width_std
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

        GROWTH_RATE = 0.5 * SEGMENT_RADIUS

        trench_length_px = self.trench_length * scale_factor
        trench_width_px = self.trench_width * scale_factor
        SUB_STEPS = self.substeps

        cell_config = CellConfig(
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
        )

        physics_config = PhysicsConfig(
            ITERATIONS=self.phys_iters * 8,
            DAMPING=0.5,
            GRAVITY=(0, self.gravity),
        )

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
                if not cell.physics_representation.segments:
                    continue
                pos_y = cell.physics_representation.segments[0].position[1]
                if pos_y < 0 or pos_y > trench_open_end:
                    sim.colony.delete_cell(cell)

        def handle_lysis(sim):
            if self.lysis_p <= 0:
                return
            for cell in sim.colony.cells[:]:
                if len(sim.colony.cells) <= 1:
                    break
                if norm.rvs() <= norm.ppf(self.lysis_p):
                    sim.colony.delete_cell(cell)

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
            for _ in range(SUB_STEPS):
                sim.step()
            # Lysis once per output frame so probability is independent of substeps
            if self.lysis_p > 0:
                handle_lysis(sim)

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

    def draw_simulation_OPL(self, do_transformation=True, label_masks=True, return_output=False):
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
