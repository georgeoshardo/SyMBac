import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import warnings

class Lineage:
    @staticmethod
    def _iter_slot_names(cell):
        slot_names = set()
        for cls in type(cell).__mro__:
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot_name in slots:
                if slot_name in {"__dict__", "__weakref__"}:
                    continue
                slot_names.add(slot_name)
        return slot_names

    @classmethod
    def _cell_to_record(cls, cell):
        attr_names = set()
        cell_dict = getattr(cell, "__dict__", None)
        if isinstance(cell_dict, dict):
            attr_names.update(cell_dict.keys())
        attr_names.update(cls._iter_slot_names(cell))
        attr_names.update({"mask_label", "mother_mask_label", "t", "mother"})

        record = {}
        for attr_name in attr_names:
            try:
                record[attr_name] = getattr(cell, attr_name)
            except AttributeError:
                continue
        return record

    @staticmethod
    def _is_missing(value):
        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    def __init__(self, simulation):
        self.simulation = simulation

        all_cells = []
        sim_dicts = []
        for cells in self.simulation.cell_timeseries:
            for cell in cells:
                all_cells.append(cell)
                sim_dicts.append(self._cell_to_record(cell))

        df = pd.DataFrame(sim_dicts)
        lineage_df = pd.DataFrame()
        family_tree_edges = []

        if {"mask_label", "mother_mask_label"}.issubset(df.columns):
            lineage_df = df.dropna(subset=["mask_label", "mother_mask_label"])
            lineage_df = lineage_df.drop_duplicates(["mask_label", "mother_mask_label"])
            family_tree_edges.extend(
                lineage_df[["mother_mask_label", "mask_label"]].itertuples(index=False, name=None)
            )

        for cell in all_cells:
            mother = getattr(cell, "mother", None)
            mother_mask_label = getattr(mother, "mask_label", None) if mother is not None else None
            mask_label = getattr(cell, "mask_label", None)
            if self._is_missing(mother_mask_label) or self._is_missing(mask_label):
                continue
            family_tree_edges.append((mother_mask_label, mask_label))

        if family_tree_edges:
            unique_edges = list(dict.fromkeys(family_tree_edges))
            self.family_tree_edgelist = np.array(unique_edges, dtype=object)
        else:
            self.family_tree_edgelist = np.empty((0, 2), dtype=object)

        self.all_cell_data_df = lineage_df
        self.family_tree_graph = nx.from_edgelist(self.family_tree_edgelist, nx.DiGraph)

        self.temporal_lineage_graph = nx.DiGraph()

        for cell in all_cells:
            mask_label = getattr(cell, "mask_label", None)
            t = getattr(cell, "t", None)
            if self._is_missing(mask_label) or self._is_missing(t):
                continue
            self.temporal_lineage_graph.add_node((mask_label, t), cell=cell)

        for node in list(self.temporal_lineage_graph.nodes):
            exp_node = (node[0], node[1]-1)
            if exp_node in self.temporal_lineage_graph.nodes:
                self.temporal_lineage_graph.add_edge(exp_node, node)

        for node in list(self.temporal_lineage_graph.nodes):
            cell = self.temporal_lineage_graph.nodes[node]["cell"]
            mother = getattr(cell, "mother", None)
            mother_mask_label = getattr(mother, "mask_label", None) if mother is not None else None
            if self._is_missing(mother_mask_label):
                mother_mask_label = getattr(cell, "mother_mask_label", None)

            if self._is_missing(mother_mask_label):
                continue

            exp_node = (mother_mask_label, node[1])
            if exp_node in self.temporal_lineage_graph.nodes:
                self.temporal_lineage_graph.add_edge(exp_node, node)
            else:
                warnings.warn(
                    f"Skipping lineage edge for cell {node}: expected mother temporal node {exp_node} is missing.",
                    RuntimeWarning,
                )

    def plot_family_tree(self):
        pos = nx.nx_agraph.graphviz_layout(self.family_tree_graph, prog="twopi")
        plt.figure(figsize=(8, 8))
        nx.draw(self.family_tree_graph, pos, node_size=20, alpha=1, node_color="blue", with_labels=True)
        plt.axis("equal")
        plt.show()

    def _resolve_lineage_root(self, mask_label):
        if self._is_missing(mask_label):
            return None
        if mask_label not in self.family_tree_graph:
            return mask_label

        current = mask_label
        visited = set()
        while True:
            predecessors = list(self.family_tree_graph.predecessors(current))
            if not predecessors:
                return current
            # There should only be one parent in a lineage tree. Keep this
            # deterministic even if malformed input introduces more.
            next_parent = sorted(predecessors)[0]
            if next_parent in visited:
                return current
            visited.add(next_parent)
            current = next_parent

    def to_geff(
        self,
        store,
        pix_mic_conv=None,
        resize_amount=None,
        frame_range=None,
        zarr_format=2,
        overwrite=False,
    ):
        """
        Export the temporal lineage graph to GEFF format.

        Parameters
        ----------
        store : str or MutableMapping
            GEFF output store path or zarr store object.
        pix_mic_conv : float or None, optional
            Microns per pixel conversion. If None, attempts to read from
            ``self.simulation.pix_mic_conv``.
        resize_amount : float or None, optional
            Upscaling factor used during simulation/rendering. If None, attempts
            to read from ``self.simulation.resize_amount``.
        frame_range : tuple(int, int) or None, optional
            Optional ``(start, end)`` frame window (inclusive-exclusive).
        zarr_format : int, optional
            Zarr format version passed through to ``geff.write``.
        overwrite : bool, optional
            Whether to allow overwriting an existing store.
        """
        try:
            import geff
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "GEFF export requires the 'geff' package. Install with: pip install \"SyMBac[geff]\""
            ) from exc

        if pix_mic_conv is None:
            pix_mic_conv = getattr(self.simulation, "pix_mic_conv", None)
        if resize_amount is None:
            resize_amount = getattr(self.simulation, "resize_amount", None)

        if resize_amount in (None, 0):
            spatial_scale = 1.0
        elif pix_mic_conv is None:
            spatial_scale = 1.0
        else:
            spatial_scale = float(pix_mic_conv) / float(resize_amount)

        start_frame = None
        end_frame = None
        if frame_range is not None:
            if not isinstance(frame_range, (tuple, list)) or len(frame_range) != 2:
                raise ValueError("frame_range must be a 2-tuple/list: (start, end).")
            start_frame, end_frame = frame_range
            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = np.inf
            if end_frame <= start_frame:
                raise ValueError("frame_range must satisfy end > start.")

        def _frame_selected(t):
            if start_frame is not None and t < start_frame:
                return False
            if end_frame is not None and t >= end_frame:
                return False
            return True

        selected_nodes = [
            node
            for node in self.temporal_lineage_graph.nodes
            if _frame_selected(node[1])
        ]
        selected_nodes.sort(key=lambda node: (node[1], node[0]))

        node_lookup = {}
        geff_graph = nx.DiGraph()

        for geff_node_id, node in enumerate(selected_nodes):
            mask_label, t = node
            cell = self.temporal_lineage_graph.nodes[node].get("cell")
            if cell is None:
                continue

            position = getattr(cell, "position", None)
            if position is not None:
                x_px = float(position[0])
                y_px = float(position[1])
            else:
                segment_positions = getattr(cell, "segment_positions", None)
                if segment_positions is not None and len(segment_positions) > 0:
                    mean_xy = np.mean(segment_positions, axis=0)
                    x_px = float(mean_xy[0])
                    y_px = float(mean_xy[1])
                else:
                    x_px = np.nan
                    y_px = np.nan

            lineage_root = self._resolve_lineage_root(mask_label)
            generation = getattr(cell, "generation", 0)
            generation = int(generation) if not self._is_missing(generation) else 0

            geff_graph.add_node(
                geff_node_id,
                t=int(t),
                x=float(x_px * spatial_scale) if np.isfinite(x_px) else np.nan,
                y=float(y_px * spatial_scale) if np.isfinite(y_px) else np.nan,
                seg_id=int(mask_label),
                tracklet_id=int(mask_label),
                lineage_id=int(lineage_root) if not self._is_missing(lineage_root) else -1,
                length=float(getattr(cell, "length", np.nan)),
                width=float(getattr(cell, "width", np.nan)),
                angle=float(getattr(cell, "angle", np.nan)),
                generation=generation,
            )
            node_lookup[node] = geff_node_id

        for source, target in self.temporal_lineage_graph.edges:
            if source in node_lookup and target in node_lookup:
                geff_graph.add_edge(node_lookup[source], node_lookup[target])

        # geff backend kwargs vary across versions; newer docs include
        # track-node mapping kwargs that older versions reject.
        try:
            geff.write(
                geff_graph,
                store,
                axis_names=["t", "y", "x"],
                axis_types=["time", "space", "space"],
                zarr_format=zarr_format,
                overwrite=overwrite,
                track_node_props={"tracklet": "tracklet_id", "lineage": "lineage_id"},
            )
        except TypeError as exc:
            if "track_node_props" not in str(exc):
                raise
            geff.write(
                geff_graph,
                store,
                axis_names=["t", "y", "x"],
                axis_types=["time", "space", "space"],
                zarr_format=zarr_format,
                overwrite=overwrite,
            )
