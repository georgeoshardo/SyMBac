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
