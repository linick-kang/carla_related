import glob
import os
import sys

"""
v1 for monthly report - scene2.py
"""

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import math
import numpy as np
import open3d as o3d

localhost = '127.0.0.1'
port = 2000

def visualize(data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([pcd])	            

def local_transform(xyz):
    # unreal to normal orientation transform
    tm = np.zeros((3,3))
    tm[1,0] = tm[2,2] = -1
    tm[0,1] = 1
    transformed = xyz @ tm
    return transformed

def world_transform(xyz, dx, dy, dz, pitch, roll, yaw, scale_x = 1, scale_y = 1, scale_z = 1):
    # sensor to world coodinate transform
    # xyz is expected to as (n, 3) shape numpy array
    # pitch, roll, yaw in degrees
    add = np.ones((xyz.shape[0],1))
    origin = np.hstack((xyz,add))
    tm = np.zeros((4,4))
    tm[0,3] = dx
    tm[1,3] = dy
    tm[2,3] = dz
    tm[3,3] = 1
    cy = math.cos(np.radians(yaw))
    sy = math.sin(np.radians(yaw))
    cr = math.cos(np.radians(roll))
    sr = math.sin(np.radians(roll))
    cp = math.cos(np.radians(pitch))
    sp = math.sin(np.radians(pitch))
    tm[0, 0] = scale_x * (cp * cy)
    tm[0, 1] = scale_x * (cy * sp * sr - sy * cr)
    tm[0, 2] = scale_x * (cy * sp * cr + sy * sr)
    tm[1, 0] = scale_y * (sy * cp)
    tm[1, 1] = scale_y * (sy * sp * sr + cy * cr)
    tm[1, 2] = scale_y * (sy * sp * cr - cy * sr)
    tm[2, 0] = scale_z * (-sp)
    tm[2, 1] = scale_z * (cp * sr)
    tm[2, 2] = scale_z * (cp * cr)
    transformed = origin @ tm
    new_xyz = transformed[:,:3]
    return new_xyz

def main():
    actor_list = []
    sensor_list = []
    dlist = [None] * 3

    try:
        client = carla.Client(localhost, port)
        client.set_timeout(2.0)
        world = client.get_world()

        settings = world.get_settings()
        settings.synchronous_mode = True
        world.apply_settings(settings)

        view1 = carla.Transform(carla.Location(x=88, y=-145, z=10), carla.Rotation(pitch=0, yaw=120, roll=0))
        topview = carla.Transform(carla.Location(x=86, y=-143, z=28), carla.Rotation(pitch=-70, yaw=110, roll=0))
        world.get_spectator().set_transform(topview)
        t_start = time.time()

        """
        Weather
        """
        
#        weather = world.get_weather()
#        weather.cloudiness = 0
#        weather.precipitation = 0
#        weather.precipitation_deposits = 0
#        weather.wind_intensity = 0
#        weather.sun_azimuth_angle = 0
#        weather.sun_altitude_angle = 60
#        weather.fog_density = 65
#        weather.fog_distance = 10
#        world.set_weather(weather)
        
        """
        Target
        """        

        bplib = world.get_blueprint_library()
        #bp = random.choice(blueprint_library.filter('vehicle'))
        
        bp1 = bplib.filter("model3")[0]
        
        # modify blueprint
        if bp1.has_attribute('color'):
            color = random.choice(bp1.get_attribute('color').recommended_values)
            bp1.set_attribute('color', color)
        
        
        location1 = carla.Transform(carla.Location(x=86, y=-138, z=10), carla.Rotation(pitch=0, yaw=270, roll=0))
        target1 = world.spawn_actor(bp1, location1)
        actor_list.append(target1)
        
        location2 = carla.Transform(carla.Location(x=84, y=-143, z=10), carla.Rotation(pitch=0, yaw=0, roll=0))
        ped_bps = bplib.filter("walker.*")
        random.seed(165)
        target2 = world.try_spawn_actor(random.choice(ped_bps), location2)
        actor_list.append(target2)
        
        """
        Sensor
        """        


        # pre-define positions
#        lvp1 = carla.Transform(carla.Location(x=92.3, y=-156.9, z=10.03), carla.Rotation(pitch=1.9, yaw=10, roll=1))
#        lvp2 = carla.Transform(carla.Location(x=74.8, y=-155.6, z=10.17), carla.Rotation(pitch=3, yaw=55, roll=0.3))
#        lvp3 = carla.Transform(carla.Location(x=72.6, y=-114.8, z=10.08), carla.Rotation(pitch=1.1, yaw=190, roll=0.5))
        lvp1 = carla.Transform(carla.Location(x=92.3, y=-156.9, z=10), carla.Rotation(pitch=3.9, yaw=10, roll=3.3))
        lvp2 = carla.Transform(carla.Location(x=74.8, y=-155.6, z=10), carla.Rotation(pitch=6, yaw=55, roll=4.3))
        lvp3 = carla.Transform(carla.Location(x=72.6, y=-114.8, z=10), carla.Rotation(pitch=5.1, yaw=190, roll=3.5))
        lvp_list = [lvp1, lvp2, lvp3]

        # --------------
        # Lidar blue print
        # --------------
#        """
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',str(16))
        lidar_bp.set_attribute('points_per_second',str(50000))
        lidar_bp.set_attribute('rotation_frequency',str(100))
        lidar_bp.set_attribute('range',str(100))
        lidar_bp.set_attribute('lower_fov',str(-15))
        lidar_bp.set_attribute('upper_fov',str(15))
#        lidar_bp.set_attribute('sensor_tick', str(0.1))
        
        lidar1 = None
        lidar2 = None
        lidar3 = None
        
        lidar1 = world.spawn_actor(lidar_bp, lvp1)
        lidar2 = world.spawn_actor(lidar_bp, lvp2)
        lidar3 = world.spawn_actor(lidar_bp, lvp3)
        
        sensor_list.append(lidar1)
        sensor_list.append(lidar2)
        sensor_list.append(lidar3)
        
        #lidar.listen(lambda point_cloud: point_cloud.save_to_disk('sensor_result/lidar/%.6d.ply' % point_cloud.frame))
        def get_points1(data):
            t_pass = time.time() - t_start
            header = np.array([1, t_pass])
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
#            points = local_transform(points)
            dlist[0] = [header, points]
        def get_points2(data):
            t_pass = time.time() - t_start
            header = np.array([2, t_pass])
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
#            points = local_transform(points)
            dlist[1] = [header, points]
        def get_points3(data):
            t_pass = time.time() - t_start
            header = np.array([3, t_pass])
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
#            points = local_transform(points)
            dlist[2] = [header, points]

            
        lidar1.listen(get_points1)
        lidar2.listen(get_points2)
        lidar3.listen(get_points3)

#        """        
 
        """
        Simulation pseudo-realtime
        """
        sensor_interval = 0.1
        i = 0
        while True:
            world.tick()
            t_pass = time.time() - t_start
            if t_pass >= i * sensor_interval:
                d1 = dlist[0]
                d2 = dlist[1]
                d3 = dlist[2]
                i += 1
            if i == 50:
                plot_time = d1[0][1]
                pcl_list = [d1, d2, d3]
                import pickle as pkl
                output = open('sensor_data_mr1.pkl', 'wb')
                pkl.dump(pcl_list, output)
                output.close()
                        
            if time.time() - t_start >= 8:
                        
                def integrate_point_cloud(data_list, transform_list):
                    xyz_full = np.zeros((0,3))
                    for data in data_list:
                        lidar_id = data[0][0]
                        lidar_transform = transform_list[int(lidar_id - 1)]
                        dx = lidar_transform.location.x
                        dy = lidar_transform.location.y
                        dz = lidar_transform.location.z
                        pitch = lidar_transform.rotation.pitch
                        roll = lidar_transform.rotation.roll
                        yaw = lidar_transform.rotation.yaw
                        xyz_local = data[1]
                        xyz_world = world_transform(xyz_local, dx, dy, dz, pitch, roll, yaw)
                        xyz_full = np.vstack((xyz_full, xyz_world))
                    return xyz_full
                        
#                multi_lidar_res = integrate_point_cloud(pcl_list, lvp_list) # integrated 3-lidar point cloud in one single frame
#                visualize(multi_lidar_res)
                break


    finally:

            
        for actor in actor_list:
            actor.destroy()
        for sensor in sensor_list:
            sensor.stop()
            sensor.destroy()
        print("All cleaned up!")
        
if __name__ == '__main__':
    main()