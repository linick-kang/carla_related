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

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=translation)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

def visualize_multiple(mdata):
    np.random.seed(14)
    pcd_list = []
    for data in mdata:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)
        pcd.paint_uniform_color(list(np.random.rand(3)))
        pcd_list.append(pcd)
    o3d.visualization.draw_geometries(pcd_list)
    
def visualize(data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([pcd])	            

def world_transform(xyz, dx, dy, dz, yaw, pitch, roll, unreal = True):
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
    if not unreal:
        rm_local = np.eye(3)

    rotated = (rm_local @ rm_world @ xyz.T).T
    translated = rotated + np.array([[dx, dy, dz]])
    # translated[:,1] *= -1
    return translated

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

# pre-define positions
lvp1 = carla.Transform(carla.Location(x=92.3, y=-156.9, z=10), carla.Rotation(pitch=3.9, yaw=10, roll=3.3))
lvp2 = carla.Transform(carla.Location(x=74.8, y=-155.6, z=10), carla.Rotation(pitch=6, yaw=55, roll=4.3))
lvp3 = carla.Transform(carla.Location(x=72.6, y=-114.8, z=10), carla.Rotation(pitch=5.1, yaw=190, roll=3.5))
lvp_list = [lvp1, lvp2, lvp3]

pkl_file = open('sensor_data_mr1.pkl', 'rb')
data = pkl.load(pkl_file)
pkl_file.close()

w_axis = np.zeros((150,3))
w_axis[:50,0] = np.arange(1,51)/10
w_axis[50:100,1] = np.arange(1,51)/12
w_axis[100:150,2] = np.arange(1,51)/15

def integrate_point_cloud(data_list, transform_list):
    xyz_full = np.zeros((0,3))
    for data, lidar_transform in zip(data_list, lvp_list):
        dx =  -lidar_transform.location.x
        dy =  lidar_transform.location.y
        dz =  lidar_transform.location.z
#        print("\n dx {}, dy {}, dz {}".format(dx, dy, dz))
        pitch = lidar_transform.rotation.pitch
        roll = lidar_transform.rotation.roll
        yaw = lidar_transform.rotation.yaw
        roll = 0
        pitch = 0
        # dwa = np.vstack((data[1],axis))
        dwa = data[1]
        xyz_world = world_transform(dwa, dx, dy, dz, yaw, pitch, roll)
        xyz_full = np.vstack((xyz_full,xyz_world))
    # xyz_full = np.vstack((xyz_full,w_axis))
    return xyz_full

def visualize_pcd_bb(pcd_info,bb_info):
    np.random.seed(2)
    pcd_list = []
    # for data in pcd_info:
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(data)
    #     pcd.paint_uniform_color(list(np.random.rand(3)))
    #     pcd_list.append(pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_info)
    # pcd.paint_uniform_color(list(np.random.rand(3)))
    pcd_list.append(pcd)
    bb_list = []
    for target in bb_info:
        ex,ey,ez,transform = target
        l, w, h = 2*ex, 2*ey, 2*ez
        # #bb center at object box center
        # bb_local = np.array([[-l/2,-w/2,-h/2],[l/2,-w/2,-h/2],[-l/2,w/2,-h/2],[l/2,w/2,-h/2],
        #                      [-l/2,-w/2,h/2],[l/2,-w/2,h/2],[-l/2,w/2,h/2],[l/2,w/2,h/2]])
        #bb center at object ground center
        bb_local = np.array([[-l/2,-w/2,0],[l/2,-w/2,0],[-l/2,w/2,0],[l/2,w/2,0],
                             [-l/2,-w/2,h],[l/2,-w/2,h],[-l/2,w/2,h],[l/2,w/2,h]])
        pitch = transform.rotation.pitch
        roll = transform.rotation.roll
        yaw = transform.rotation.yaw
        dx = -transform.location.x
        dy = transform.location.y
        dz = transform.location.z
        bb_world = world_transform(bb_local, dx, dy, dz, yaw, pitch, roll, False)
        points = bb_world.tolist()
        lines = [[0,1],[0,2],[1,3],[2,3],
                 [4,5],[4,6],[5,7],[6,7],
                 [0,4],[1,5],[2,6],[3,7]]
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bb_list.append(line_set)
    plot_list = pcd_list + bb_list
    o3d.visualization.draw_geometries(plot_list)

location1 = carla.Transform(carla.Location(x=86, y=-138, z=8.02), carla.Rotation(pitch=0, yaw=270, roll=0))
bb1 = carla.BoundingBox(carla.Location(x=0.030725, y=-0.005984, z=0.747461), carla.Vector3D(x=2.404411, y=1.084753, z=0.761591))
location2 = carla.Transform(carla.Location(x=84, y=-143, z=8.02), carla.Rotation(pitch=0, yaw=0, roll=0))
bb2 = carla.BoundingBox(carla.Location(x=0.000000, y=0.000000, z=0.000000), carla.Vector3D(x=0.340000, y=0.340000, z=0.930000))
target1 = [bb1.extent.x, bb1.extent.y, bb1.extent.z, location1]
target2 = [bb2.extent.x, bb2.extent.y, bb2.extent.z, location2]
targets = [target1, target2]

multi_lidar_res = integrate_point_cloud(data, lvp_list) # integrated 3-lidar point cloud in one single frame
# visualize_pcd_bb(multi_lidar_res, targets)
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
                "eps":2,
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

from scipy.spatial import ConvexHull

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval

def create_o3d_circle(xy, color, z=8.5, radius = 0.45):
    x,y = xy
    
    num_edge = 12
    
    thetas = np.arange(num_edge)/num_edge * np.pi * 2
    xyz = np.vstack((np.cos(thetas) * radius + x, np.sin(thetas) * radius + y, np.ones(num_edge) * z)).T
    points = xyz.tolist()
    
    lines = [[i,i+1] for i in range(num_edge-1)] + [[num_edge-1,0]]
    
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines),
                )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_mesh = LineMesh(points, lines, colors, radius=0.02)
    line_mesh_geoms = line_mesh.cylinder_segments
    return line_mesh_geoms

def create_o3d_rect(corners_2d, color, z = 8.5):
    points = np.hstack((corners_2d,np.ones((4,1))*z)).tolist()
    lines = [[0,1],[1,2],[2,3],[3,0]]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines),
                )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_mesh = LineMesh(points, lines, colors, radius=0.02)
    line_mesh_geoms = line_mesh.cylinder_segments
    return line_mesh_geoms
    
target_location = []
veh_corners = []
detection_bb = []
detect_color = [0.2,0.8,0.2]
detect_z = 8.5
i = 1
if len(ped_clusters_2d) > 0:
    for tar in ped_clusters_2d:
        # target_id, center_xy, category(1:ped, 2:veh)
        corners = minimum_bounding_rectangle(tar)
        loc_xy = np.mean(corners,axis=0)
        target_location.append([i, loc_xy, 1])
        detection_bb.extend(create_o3d_circle(loc_xy, detect_color, detect_z))
        i += 1
        
if len(veh_clusters_2d) > 0:
    for tar in veh_clusters_2d:
        corners = minimum_bounding_rectangle(tar)
        loc_xy = np.mean(corners,axis=0)
        target_location.append([i, loc_xy, 2])
        veh_corners.append([i, corners])
        detection_bb.extend(create_o3d_rect(corners, detect_color, detect_z))
        i += 1


        
def visualize_detection(target_pcd,bb3d_info,detection):
    # np.random.seed(456)
    # target_num = len(target_pcd)
    color_list = [[0.2,0.2,1],[0,0,0]]
    # color_list = np.random.rand(target_num,3).tolist()
    pcd_list = []
    for i, data in enumerate(target_pcd):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)
        pcd.paint_uniform_color(color_list[i])
        pcd_list.append(pcd)

    bb3d_list = []
    for target in bb3d_info:
        ex,ey,ez,transform = target
        l, w, h = 2*ex, 2*ey, 2*ez
        # #bb center at object box center
        # bb_local = np.array([[-l/2,-w/2,-h/2],[l/2,-w/2,-h/2],[-l/2,w/2,-h/2],[l/2,w/2,-h/2],
        #                      [-l/2,-w/2,h/2],[l/2,-w/2,h/2],[-l/2,w/2,h/2],[l/2,w/2,h/2]])
        #bb center at object ground center
        bb_local = np.array([[-l/2,-w/2,0],[l/2,-w/2,0],[-l/2,w/2,0],[l/2,w/2,0],
                             [-l/2,-w/2,h],[l/2,-w/2,h],[-l/2,w/2,h],[l/2,w/2,h]])
        pitch = transform.rotation.pitch
        roll = transform.rotation.roll
        yaw = transform.rotation.yaw
        dx = -transform.location.x
        dy = transform.location.y
        dz = transform.location.z
        bb_world = world_transform(bb_local, dx, dy, dz, yaw, pitch, roll, False)
        points = bb_world.tolist()
        lines = [[0,1],[0,2],[1,3],[2,3],
                 [4,5],[4,6],[5,7],[6,7],
                 [0,4],[1,5],[2,6],[3,7]]
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_mesh = LineMesh(points, lines, colors, radius=0.02)
        line_mesh_geoms = line_mesh.cylinder_segments
        bb3d_list.extend(line_mesh_geoms)
    
    o3d.visualization.RenderOption.point_size = 20
    o3d.visualization.RenderOption.show_coordinate_frame = True
    # plot_list = pcd_list + bb3d_list
    plot_list = pcd_list + bb3d_list + detection
    o3d.visualization.draw_geometries(plot_list)
    
target_pcd = veh_clusters_3d + ped_clusters_3d
visualize_detection(target_pcd, targets, detection_bb)
