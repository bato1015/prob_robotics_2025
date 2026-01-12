import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.special import digamma, logsumexp

pcd = o3d.io.read_point_cloud("1109_ver0_0_4_216westimate_bo_x_z.pcd")
points_3d = np.asarray(pcd.points)
points_xz = points_3d[:, [0, 2]]
geoms = []
geoms.append(pcd)
o3d.visualization.draw_geometries(geoms)
# Cluster = 100

# N,D = points_xz.shape
# r = rng.random((N, Cluster))

# N_sum = r.sum(axis=0)
