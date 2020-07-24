"""
scene2 for monthly report graph/situation
"""

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

def main():
    actor_list = []
    sensor_list = []
    dlist = [None] * 3

    try:
        client = carla.Client(localhost, port)
        client.set_timeout(2.0)
        world = client.get_world()

        # settings = world.get_settings()
        # settings.synchronous_mode = True
        # world.apply_settings(settings)

        view1 = carla.Transform(carla.Location(x=88, y=-145, z=10), carla.Rotation(pitch=0, yaw=120, roll=0))
        topview = carla.Transform(carla.Location(x=85, y=-155, z=18), carla.Rotation(pitch=-40, yaw=90, roll=0))
        world.get_spectator().set_transform(topview)
        t_start = time.time()

        """
        Weather
        """
        
        weather = world.get_weather()
        weather.cloudiness = 0
        weather.precipitation = 0
        weather.precipitation_deposits = 0
        weather.wind_intensity = 0
        weather.sun_azimuth_angle = 80
        weather.sun_altitude_angle = 40
#        weather.fog_density = 65
#        weather.fog_distance = 10
        world.set_weather(weather)
        
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
 
        # location3 = carla.Transform(carla.Location(x=0, y=0, z=10), carla.Rotation(pitch=0, yaw=0, roll=0))
        # # ped_bps = bplib.filter("cola")[0]
        # target3 = world.try_spawn_actor(bp1, location3)
        # actor_list.append(target3)        
 
        """
        Simulation pseudo-realtime
        """
        while True:
            try:
                time.sleep(0.1)
            except KeyboardInterrupt:
                print('End')
                break
            else:
                #print(t.location)
                pass
    finally:
        for actor in actor_list:
            print(actor.get_transform())
            print(actor.bounding_box)
            actor.destroy()     
        print("All cleaned up!")
        
if __name__ == '__main__':
    main()