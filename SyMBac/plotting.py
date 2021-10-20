import numpy as np
import matplotlib.pyplot as plt

def matplot_scene(a, cells, savedir):
    if a%1 == 0:
        for cell in cells:
            vertices = cell.get_vertex_list()
            vertices = np.array(vertices)
            plt.plot(vertices[:,0], vertices[:,1],c="k")
            #centroid = cell.position
            #plt.scatter(centroid[0],centroid[1],s=100)
        plt.ylim(0,800)
        plt.xlim(0,800)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(savedir+"/image_{}.pdf".format(str(a).zfill(3)),dpi=100)
        plt.clf()

