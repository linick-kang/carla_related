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

test_pipeline = """
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
                "limits":"X[-93:-72], Y[-157:-114], Classification[1:1]"
            },
            {
                "type":"filters.dbscan",
                "min_points":3,
                "eps":2,
                "dimensions":"X,Y,Z"
            }
    ]
}
"""

p = pdal.Pipeline(test_pipeline, arrays=[point_data,])
p.validate()
count = p.execute()
result = p.arrays[0]
xyz = np.vstack((result['X'],result['Y'],result['Z'],result['ClusterID'])).T
labels = np.unique(result['ClusterID'])
clusters = []
for lid in labels:
    clusters.append(xyz[xyz[:,3] == lid][:,:3])
visualize_cluster(clusters)
# visualize(xyz)
# xyz = np.vstack((result['X'],result['Y'],result['Z'],result['Classification'])).T
# non_ground = xyz[xyz[:,3] == 1][:,:3]
# ground = xyz[xyz[:,3] == 2][:,:3]
# visualize_pair([ground, non_ground])