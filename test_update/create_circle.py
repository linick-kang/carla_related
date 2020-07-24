import numpy as np
import open3d as o3d


z = 0
radius = 1
num_edge = 12

thetas = np.arange(num_edge)/num_edge * np.pi * 2
xyz = np.vstack((np.cos(thetas) * radius, np.sin(thetas) * radius, np.ones(num_edge) * z)).T
points = xyz.tolist()

lines = [[i,i+1] for i in range(num_edge-1)] + [[num_edge-1,0]]

colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([line_set])