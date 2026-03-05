from types import SimpleNamespace

from SyMBac.config_models import (
    DatasetOutputConfig,
    RandomDatasetPlan,
    RenderConfig,
    SimulationCellSpec,
    SimulationGeometrySpec,
    SimulationPhysicsSpec,
    SimulationRuntimeSpec,
    SimulationSpec,
    TimeseriesDatasetPlan,
)
from SyMBac.napari.controllers.workflow_controller import WorkflowController
from SyMBac.napari.state import NapariSessionState


class _SimulationServiceStub:
    def __init__(self):
        self.ran = False
        self.drew = False

    def build(self, spec):
        return SimpleNamespace(spec=spec)

    def run(self, simulation, show_window=False):
        simulation.show_window = show_window
        self.ran = True
        return simulation

    def draw_opl(self, simulation, do_transformation=False, label_masks=True):
        simulation.OPL_scenes = [1]
        simulation.masks = [2]
        self.drew = True
        return simulation


class _RendererServiceStub:
    def build_psf(self, params):
        return {"psf": params}

    def build_camera(self, params):
        return {"camera": params}

    def build_renderer(self, **kwargs):
        return SimpleNamespace(kwargs=kwargs)

    def preview_frame(self, renderer, frame_index, config):
        return SimpleNamespace(image=frame_index, mask=config)


class _RegionServiceStub:
    def auto_segment(self, renderer, classes=3, cells="dark"):
        return {"media": [True], "cell": [False], "device": [False]}

    def apply_to_renderer(self, renderer, masks):
        renderer.masks_set = masks


class _ExportServiceStub:
    def export_dataset(self, renderer, plan, output, base_config, seed=None):
        return {
            "renderer": renderer,
            "plan": plan.kind,
            "output": output.save_dir,
            "seed": seed,
            "base": base_config.kind,
        }

    def export_batch_timeseries(self, **kwargs):
        return [{"ok": True, "count": len(kwargs["variants"])}]


def _spec(tmp_path):
    return SimulationSpec(
        geometry=SimulationGeometrySpec(
            trench_length=14.0,
            trench_width=1.5,
            pix_mic_conv=0.065,
            resize_amount=1,
        ),
        cell=SimulationCellSpec(
            cell_max_length=6.0,
            cell_width=1.0,
            max_length_std=0.0,
            width_std=0.0,
            lysis_p=0.0,
        ),
        physics=SimulationPhysicsSpec(gravity=0.0, phys_iters=1),
        runtime=SimulationRuntimeSpec(sim_length=2, substeps=1, save_dir=str(tmp_path / "sim")),
    )


def test_controller_main_flow(tmp_path):
    state = NapariSessionState()
    controller = WorkflowController(
        state=state,
        simulation_service=_SimulationServiceStub(),
        renderer_service=_RendererServiceStub(),
        region_service=_RegionServiceStub(),
        export_service=_ExportServiceStub(),
    )

    spec = _spec(tmp_path)
    controller.set_simulation_spec(spec)
    sim = controller.run_simulation(show_window=False)
    assert sim.spec == spec

    controller.draw_opl()
    assert state.simulation.OPL_scenes == [1]

    controller.set_real_image([[1]])
    renderer = controller.build_renderer(psf_params={"radius": 1}, camera_params={"baseline": 1})
    assert renderer.kwargs["real_image"] == [[1]]

    regions = controller.auto_segment_regions(classes=3, cells="dark")
    assert regions["media"] == [True]

    preview = controller.preview_frame(frame_index=3, config=RenderConfig())
    assert preview.image == 3

    meta = controller.export_dataset(
        plan=RandomDatasetPlan(n_samples=1),
        output=DatasetOutputConfig(save_dir=str(tmp_path / "out")),
        base_config=RenderConfig(),
        seed=7,
    )
    assert meta["seed"] == 7

    batch = controller.export_batch_timeseries(
        variants=[{"cell_max_length": 6.1, "cell_width": 1.0}],
        plan=TimeseriesDatasetPlan(n_series=1),
        output=DatasetOutputConfig(save_dir=str(tmp_path / "batch_out")),
        base_config=RenderConfig(),
        seed=9,
    )
    assert batch[0]["count"] == 1
