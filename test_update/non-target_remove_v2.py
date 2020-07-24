import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import pickle as pkl
import carla
import math
import numpy as np
import open3d as o3d

def visualize_cluster(mdata):
    np.random.seed(14)
    pcd_list = []
    for i, data in enumerate(mdata):
        if i == 0:
            color = [0, 0, 0]
        else:
            color = list(np.random.rand(3))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)
        pcd.paint_uniform_color(color)
        pcd_list.append(pcd)
    o3d.visualization.draw_geometries(pcd_list)

def visualize_multiple(mdata):
    np.random.seed(14)
    pcd_list = []
    for data in mdata:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)
        pcd.paint_uniform_color(list(np.random.rand(3)))
        pcd_list.append(pcd)
    o3d.visualization.draw_geometries(pcd_list)

def visualize_pair(pdata):
    color_list = [[0,0,1],[0,0,0]]
    pcd_list = []
    for i, data in enumerate(pdata):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)
        pcd.paint_uniform_color(color_list[i])
        pcd_list.append(pcd)
    o3d.visualization.draw_geometries(pcd_list)    
    
def visualize(data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([pcd])	            

def world_transform(xyz, dx, dy, dz, yaw, pitch, roll):
    # sensor to world coodinate transform
    # xyz is expected to as (n, 3) shape numpy array
    # pitch, roll, yaw in degrees
#    rm = np.zeros((3,3))
    cy = math.cos(np.radians(yaw))
    sy = math.sin(np.radians(yaw))
    cp = math.cos(np.radians(pitch))
    sp = math.sin(np.radians(pitch))
    cr = math.cos(np.radians(roll))
    sr = math.sin(np.radians(roll))

    #alpha yaw, beta pitch, gamma roll 
#    rm[0, 0] = cy * cp
#    rm[0, 1] = cy * sp * sr - sy * cr
#    rm[0, 2] = cy * sp * cr + sy * sr
#    rm[1, 0] = sy * cp
#    rm[1, 1] = sy * sp * sr + cy * cr
#    rm[1, 2] = sy * sp * cr - cy * sr
#    rm[2, 0] = -sp
#    rm[2, 1] = cp * sr
#    rm[2, 2] = cp * cr
    
    rm_yaw = np.array([[cy,-sy, 0],
                       [sy, cy, 0],
                       [0,  0,  1]])
    rm_pitch = np.array([[cp, 0, sp],
                         [0,  1,  0],
                         [-sp, 0, cp]])
    rm_roll = np.array([[1, 0, 0],
                        [0, cr, -sr],
                        [0, sr, cr]])
    
    rm_world =  rm_pitch @ rm_roll @ rm_yaw
    rm_local =  np.array([[0, 1, 0],
                          [1, 0, 0],
                          [0, 0, -1]])

    rotated = (rm_local @ rm_world @ xyz.T).T
    translated = rotated + np.array([[dx, dy, dz]])
    # translated[:,1] *= -1
    return translated


# pre-define positions
lvp1 = carla.Transform(carla.Location(x=92.3, y=-156.9, z=10), carla.Rotation(pitch=3.9, yaw=10, roll=3.3))
lvp2 = carla.Transform(carla.Location(x=74.8, y=-155.6, z=10), carla.Rotation(pitch=6, yaw=55, roll=4.3))
lvp3 = carla.Transform(carla.Location(x=72.6, y=-114.8, z=10), carla.Rotation(pitch=5.1, yaw=190, roll=3.5))
lvp_list = [lvp1, lvp2, lvp3]

pkl_file = open('sensor_data_mr1.pkl', 'rb')
data = pkl.load(pkl_file)
pkl_file.close()

axis = np.zeros((150,3))
axis[:50,0] = np.arange(1,51)/15
axis[50:100,1] = np.arange(1,51)/20
axis[100:150,2] = np.arange(1,51)/25

w_axis = np.zeros((150,3))
w_axis[:50,0] = np.arange(1,51)/10
w_axis[50:100,1] = np.arange(1,51)/12
w_axis[100:150,2] = np.arange(1,51)/15

def integrate_point_cloud2(data_list, transform_list):
    xyz_full = np.zeros((0,3))

    for data, lidar_transform in zip(data_list, lvp_list):
        dx =  -lidar_transform.location.x
        dy =  +lidar_transform.location.y
        dz =  +lidar_transform.location.z
#        print("\n dx {}, dy {}, dz {}".format(dx, dy, dz))
        pitch = lidar_transform.rotation.pitch
        roll = lidar_transform.rotation.roll
        yaw = lidar_transform.rotation.yaw
        roll = 0
        pitch = 0
        # dwa = np.vstack((data[1],axis))
        # xyz_local = local_transform(dwa)
        xyz_world = world_transform(data[1], dx, dy, dz, yaw, pitch, roll)
        xyz_full = np.vstack((xyz_full,xyz_world))
        

    # xyz_full = np.vstack((xyz_full,w_axis))
    return xyz_full
#                pcl_list = [d1, d2, d3]      
multi_lidar_res = integrate_point_cloud2(data, lvp_list) # integrated 3-lidar point cloud in one single frame
#ground, non_ground = ground_segmentation(multi_lidar_res)
#visualize_multiple([ground[:,:3], non_ground[:,:3]])
a = multi_lidar_res.copy()

import pdal

point_data = np.array(
    [(x, y, z) for x, y, z in a[:,:]],
    dtype=[('X', np.float), ('Y', np.float), ('Z', np.float)]
)

pipeline1 = """
{
    "pipeline": [
            {
                "type":"filters.smrf",
                "scalar":1.2,
                "slope":0.2,
                "threshold":0.45,
                "window":16.0
            },
            {
                "type":"filters.range",
                "limits":"X[-93:-72], Y[-157:-114]"
            },
            {
                "type":"filters.hag",
                "count":4
            },
            {
                "type":"filters.ferry",
                "dimensions":"HeightAboveGround=>H"
            },
            {
                "type":"filters.range",
                "limits":"Classification[1:1]"
            },
            {
                "type":"filters.dbscan",
                "min_points":3,
                "eps":1.5,
                "dimensions":"X,Y,Z"
            }
    ]
}
"""
p = pdal.Pipeline(pipeline1, arrays=[point_data,])
p.validate()
count = p.execute()
result = p.arrays[0]
xyz = np.vstack((result['X'],result['Y'],result['Z'],result['ClusterID'],result['H'])).T
labels = np.unique(result['ClusterID'])
ped_clusters_3d = []
veh_clusters_3d = []
env_clusters_3d = np.zeros((0,3))
ped_clusters_2d = []
veh_clusters_2d = []

for lid in labels:
    cluster = xyz[xyz[:,3] == lid]
    if np.max(cluster[:,4]) <= 3:
        cluster_xyz = cluster[:,:3]
        size = np.max(cluster_xyz, axis=0) - np.min(cluster_xyz, axis=0)
        if size[0] > 1.5 and size[1] > 3:
            veh_clusters_3d.append(cluster_xyz)
            veh_clusters_2d.append(cluster[:,:2])
        elif size[1] > 1.5 and size[0] > 3:
            veh_clusters_3d.append(cluster_xyz)
            veh_clusters_2d.append(cluster[:,:2])
        elif size[0] < 0.7 and size[1] < 0.7:
            ped_clusters_3d.append(cluster_xyz)
            ped_clusters_2d.append(cluster[:,:2])
        else:
            env_clusters_3d = np.vstack((env_clusters_3d,cluster[:,:3]))
    else:
        env_clusters_3d = np.vstack((env_clusters_3d,cluster[:,:3]))
        
visualize_cluster(ped_clusters_3d+veh_clusters_3d)
        
# import matplotlib.pyplot as plt

# p2d = clusters_2d[0]
# plt.figure()
# plt.scatter(p2d[:,0], p2d[:,1])
# plt.gca().set_aspect('equal', adjustable='box')

# def convex_hull_graham(points):
#     from functools import reduce
#     '''
#     Returns points on convex hull in CCW order according to Graham's scan algorithm. 
#     By Tom Switzer <thomas.switzer@gmail.com>.
#     '''
#     TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)

#     def cmp(a, b):
#         return (a > b) - (a < b)

#     def turn(p, q, r):
#         return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)

#     def _keep_left(hull, r):
#         while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
#             hull.pop()
#         if not len(hull) or hull[-1] != r:
#             hull.append(r)
#         return hull

#     points = sorted(points)
#     l = reduce(_keep_left, points, [])
#     u = reduce(_keep_left, reversed(points), [])
#     return l.extend(u[i] for i in range(1, len(u) - 1)) or l

# points = convex_hull_graham(p2d.tolist())
# p2d_hull = np.array(points)
# plt.figure()
# plt.scatter(p2d_hull[:,0], p2d_hull[:,1])
# plt.gca().set_aspect('equal', adjustable='box')