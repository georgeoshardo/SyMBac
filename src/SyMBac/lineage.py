import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

class Lineage:
    def __init__(self, simulation):
        self.simulation = simulation

        sim_dicts = []
        for cells in self.simulation.cell_timeseries:
            for cell in cells:
                sim_dicts.append(cell.__dict__)

        df = pd.DataFrame(sim_dicts)
        df = df.dropna(subset=["mother_mask_label"])
        df = df.drop_duplicates(["mask_label", "mother_mask_label"])
        self.family_tree_edgelist = np.array(df[["mother_mask_label", "mask_label"]])
        self.all_cell_data_df = df
        self.family_tree_graph = nx.from_edgelist(self.family_tree_edgelist, nx.DiGraph)

        self.temporal_lineage_graph = nx.DiGraph()

        for cells in self.simulation.cell_timeseries:
            for cell in cells:
                self.temporal_lineage_graph.add_node((cell.mask_label, cell.t), cell = cell)

        for node in self.temporal_lineage_graph.nodes:
            exp_node = (node[0], node[1]-1)
            if exp_node in self.temporal_lineage_graph.nodes:
                self.temporal_lineage_graph.add_edge(exp_node, node)

        for node in self.temporal_lineage_graph.nodes:
            cell = self.temporal_lineage_graph.nodes[node]["cell"]
            if cell.mother:
                if cell.just_divided:
                    exp_node = (cell.mother.mask_label, node[1])
                    assert exp_node in self.temporal_lineage_graph.nodes
                    self.temporal_lineage_graph.add_edge(exp_node, node)

    def plot_family_tree(self):
        pos = nx.nx_agraph.graphviz_layout(self.family_tree_graph, prog="twopi")
        plt.figure(figsize=(8, 8))
        nx.draw(self.family_tree_graph, pos, node_size=20, alpha=1, node_color="blue", with_labels=True)
        plt.axis("equal")
        plt.show()