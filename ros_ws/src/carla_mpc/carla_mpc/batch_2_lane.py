import glob
import os
import sys

#from ros_ws.src.lstr.lstr.utils import image

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
import numpy as np
from simple_pid import PID
import math

from agents.navigation.global_route_planner import GlobalRoutePlanner
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
import matplotlib.pyplot as plt
import cv2
import csv

try:
	import pygame
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
	import numpy as np
except ImportError:
	raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
	import queue
except ImportError:
	import Queue as queue



import glob
import os
import sys
from matplotlib import cm
import cv2


try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

import carla
import time

import random
import open3d as o3d

try:
	import queue
except ImportError:
	import Queue as queue

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import pprint
import argparse
import importlib
import sys
import numpy as np
import time

#import matplotlib
#matplotlib.use("Agg")

import matplotlib.pyplot as plt



class CarlaSyncBatch(object):
	def __init__(self, display, font, clock, fps=20, n_obs=12):
		self.traj_values = []
		self.actor_list = []
		self.obs_list = []
		self.sensors = []
		self.delta_seconds = 1/fps
		self._queues = []
		self.sensor_data = []
		self.behind_ego = [None]*n_obs
		self.path_pixels = []
		self.vis_path = True
		self.pre_x = []
		self.pre_y = []
		self.pre_psi = []
		self.pre_sel_psi = []
		self.steps = []
		self.sel_index = 0
		self.num_goal = 0
		self.throttle = 0
		self.prev_acc = 0
		self.vel = 0
		self.prev_vel = 0.0
		self.selected_pose = []
		self.prev_waypoint = None
		self.prev_pos = None

		self.display = display
		self.font = font
		self.clock = clock

		self.other_vehicles = np.zeros([8, 6])
		self.lane_y = [0, 3.5]#[10.5, 7.0, 3.5, 0]
		self.other_vehicles[:,1] = (np.hstack((self.lane_y,self.lane_y,self.lane_y,self.lane_y)))
		#self.other_vehicles[:,0] = np.array([ 40, 10,
		#                                85, 130,
		#                                130, 85,
		#                                -10, 45
		#            ])
		"""self.other_vehicles[:,0] = np.array([ -10, 15,
										35, 65,
										90, 110,
										140, 165
					])"""
		self.other_vehicles[:,0] = np.array([ -10, 15,      #Obstcale X positions
                                        35, 65,
                                        85, 110,
                                        130, 150
                    ])

		self.client = carla.Client('localhost', 2000)
		self.client.set_timeout(10.0)

		self.world = self.client.load_world('Town10HD')
		#self.world.set_weather(carla.WeatherParameters.CloudyNoon)
		#self.world = self.client.get_world()

		self.m =self. world.get_map()
		self.tm = self.client.get_trafficmanager()
		self.tm_port = self.tm.get_port()
		self.grp = GlobalRoutePlanner(self.m, 0.25)

		self.birdview_producer = BirdViewProducer(
				self.client,  # carla.Client
				target_size=PixelDimensions(width=300, height=800),
				pixels_per_meter=4,
				crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
				render_lanes_on_junctions=False
			)
		self.bv_img_x = 300
		self.bv_img_y = 800
		self.blueprint_library = self.world.get_blueprint_library()

		self.pid = PID(0.05, 0.0, 0.05)

		self.spawn_player()
		self.spawn_sensors()
		self.spawn_vehicles(n_obs)
		self.init()

	def init(self):
		self._settings = self.world.get_settings()
		self.frame = self.world.apply_settings(carla.WorldSettings(
			no_rendering_mode=False,
			synchronous_mode=True,
			fixed_delta_seconds=self.delta_seconds))

		def make_queue(register_event):
			q = queue.Queue()
			register_event(q.put)
			self._queues.append(q)

		make_queue(self.world.on_tick)
		for sensor in self.sensors:
			make_queue(sensor.listen)
	

	def tick(self, timeout):
		self.frame = self.world.tick()
		data = [self._retrieve_data(q, timeout) for q in self._queues]
		assert all(x.frame == self.frame for x in data)
		self.sensor_data = data


	def _retrieve_data(self, sensor_queue, timeout):
		while True:
			data = sensor_queue.get(timeout=timeout)
			if data.frame == self.frame:
				return data
	

	def spawn_player(self, start_pose=None):
		if start_pose == None:
			start_pose = carla.Transform()
			#sp = random.choice(self.m.get_spawn_points())
			#start_pose = sp
			#idx = random.randint(0, 50)
			#sp = sp[5]
			#print("pose = ", sp)
			start_pose.location.x = -50#-111#10.0#20.0
			start_pose.location.y = 137#40#-60.0
			start_pose.location.z = 0.25
			self.prev_pos = start_pose.location
			self.prev_waypoint = self.m.get_waypoint(start_pose.location, project_to_road=True)
			wp = self.m.get_waypoint(start_pose.location, project_to_road=True)
			start_pose = wp.transform
			start_pose.location.z = 0.25
		vehicle = self.world.spawn_actor(
			random.choice(self.blueprint_library.filter('vehicle.tesla.model3')),
			start_pose)
		self.actor_list.append(vehicle)
		vehicle.set_simulate_physics(True)
		#vehicle.set_autopilot(True)
		vehicle.set_autopilot(False)

		#self.tm.ignore_lights_percentage(vehicle,100)
		#self.tm.distance_to_leading_vehicle(vehicle,4)
		#self.tm.vehicle_percentage_speed_difference(vehicle,-50)
		self.vehicle = vehicle
	
	def spawn_vehicles(self, n):
		blueprint1 = self.blueprint_library.filter('vehicle.aud*')
		blueprint2 = self.blueprint_library.filter('vehicle.lin*')
		blueprint3 = self.blueprint_library.filter('vehicle.niss*')
		blueprint4 = self.blueprint_library.filter('vehicle.bmw*')
		blueprints = []
		for i in blueprint1:
			blueprints.append(i)
		for i in blueprint2:
			blueprints.append(i)
		for i in blueprint3:
			blueprints.append(i)
		for i in blueprint4:
			blueprints.append(i)

		blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
		for i in range(n):
			blueprint = np.random.choice(blueprints)
			blueprint.set_attribute('role_name', 'autopilot')
	
			if blueprint.has_attribute('color'):
				color = np.random.choice(blueprint.get_attribute('color').recommended_values)
				blueprint.set_attribute('color', color)

			if blueprint.has_attribute('driver_id'):
				driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
				blueprint.set_attribute('driver_id', driver_id)
			
			car = None
			while car is None:
				obs_pose = carla.Transform()
				ego_wp = self.m.get_waypoint(self.vehicle.get_location(), project_to_road=True)
				if self.other_vehicles[i][0]>0:
					obs_wp = ego_wp.next(self.other_vehicles[i][0])[0]
				else:
					obs_wp = ego_wp.previous(-self.other_vehicles[i][0])[0]
				if self.other_vehicles[i][1]>0:
					obs_wp = obs_wp.get_right_lane()
				#obs_pose.location.x = self.vehicle.get_location().x + self.other_vehicles[i][0]
				#obs_pose.location.y = self.vehicle.get_location().y + self.other_vehicles[i][1]
				#obs_pose.location.z = 0.25
				wp = self.m.get_waypoint(obs_pose.location, project_to_road=True)
				
				obs_pose = obs_wp.transform
				obs_pose.location.z = 0.25
				
				car = self.world.try_spawn_actor(blueprint, obs_pose)
				#print(obs_pose)

				self.tm.ignore_lights_percentage(car,100)
				self.tm.distance_to_leading_vehicle(car, -2.0)
				speed = [30, 40, 50]
				self.tm.vehicle_percentage_speed_difference(car,random.choice(speed))

			car.set_autopilot(False)
			self.actor_list.append(car)
			self.obs_list.append(car)

	def spawn_sensors(self):
		camera_rgb = self.world.spawn_actor(
			self.blueprint_library.find('sensor.camera.rgb'),
			carla.Transform(carla.Location(x=2.0, z=2.0), carla.Rotation(pitch=0.0)),
			attach_to=self.vehicle)
		self.actor_list.append(camera_rgb)
		self.sensors.append(camera_rgb)

		lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
		lidar_bp.set_attribute("noise_stddev", str(0.0))
		lidar_bp.set_attribute('upper_fov', str(2))
		lidar_bp.set_attribute('lower_fov', str(-30.0))
		lidar_bp.set_attribute('channels', str(64))
		lidar_bp.set_attribute('range', str(100))
		lidar_bp.set_attribute('rotation_frequency', str(60))
		lidar_bp.set_attribute('points_per_second', str(2000000))
		lidar = self.world.spawn_actor(
			lidar_bp,
			carla.Transform(carla.Location(x=0.0, z=2.4)),
			attach_to=self.vehicle)
		self.actor_list.append(lidar_bp)
		self.sensors.append(lidar)

		camera_seg = self.world.spawn_actor(
			self.blueprint_library.find('sensor.camera.semantic_segmentation'),
			carla.Transform(carla.Location(x=2.0, z=2.0), carla.Rotation(pitch=0.0)),
			attach_to=self.vehicle)
		self.actor_list.append(camera_seg)
		self.sensors.append(camera_seg)

		lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast_semantic')
		lidar_bp.set_attribute('upper_fov', str(2))
		lidar_bp.set_attribute('lower_fov', str(-30))
		lidar_bp.set_attribute('channels', str(64))
		lidar_bp.set_attribute('range', str(120))
		lidar_bp.set_attribute('rotation_frequency', str(60))
		lidar_bp.set_attribute('points_per_second', str(5_60_000))
		lidar = self.world.spawn_actor(
			lidar_bp,
			carla.Transform(carla.Location(x=0.0, z=2.4)),
			attach_to=self.vehicle)
		self.actor_list.append(lidar_bp)
		self.sensors.append(lidar)


	def should_quit(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return True
			elif event.type == pygame.KEYUP:
				if event.key == pygame.K_ESCAPE:
					return True
		return False
	
	def draw_image(self, surface, image, blend=False):
		array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
		array = np.reshape(array, (image.height, image.width, 4))
		array = array[:, :, :3]
		array = array[:, :, ::-1]
		image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
		if blend:
			image_surface.set_alpha(100)
		surface.blit(image_surface, (0, 0))


	def visualize(self):
		#self.sensor_data[2].convert(carla.ColorConverter.CityScapesPalette)
		#self.sensor_data[3].convert(carla.ColorConverter.LogarithmicDepth)
		self.draw_image(self.display, self.sensor_data[1])
		self.display.blit(
			self.font.render('Location = % 5d , % 5d ' %(self.vehicle.get_location().x, self.vehicle.get_location().y), True, (255, 255, 255)),
			(8, 10))
		v = self.vehicle.get_velocity()
		vel = ((v.x)**2 + (v.y)**2 + (v.z)**2)**0.5
		self.display.blit(
			self.font.render('Velocity = % 5d ' % vel, True, (255, 255, 255)),
			(8, 28))
		pygame.display.flip()

		self.pre_sel_psi = []
		birdview = self.birdview_producer.produce(
					agent_vehicle=self.vehicle  # carla.Actor (spawned vehicle)
		)
		# Use only if you want to visualize
		# produces np.ndarray of shape (height, width, 3)
		rgb = BirdViewProducer.as_rgb(birdview)
		if self.vis_path:
			prev = 0
			for i in range(self.num_goal):
				#path_pixels = np.array(self.pre_x[prev+2:prev+100],self.pre_x[prev+2:prev+100] )
				path_x = self.pre_x[prev+0:prev+self.steps[i]]
				path_y = self.pre_y[prev+0:prev+self.steps[i]]
				path_psi = self.pre_psi[prev+0:prev+self.steps[i]]
				#poses = self.centerline_to_global(self.pre_x[prev+2:prev+100], self.pre_y[prev+2:prev+100],
				#                 self.pre_psi[prev+2:prev+100])
				poses = self.centerline_to_global(path_x, path_y, path_psi, i)
				path_pixels = self.path_to_pixel(poses)
				#print(path_pixels)
				if i == self.sel_index:
					#ind = 0
					for angle in poses:
						#print(angle.rotation.yaw, end="  ")
						"""if angle.rotation.yaw<0 and angle.rotation.yaw>-360:
							angle.rotation.yaw = 360 + angle.rotation.yaw
						if angle.rotation.yaw<-360 and angle.rotation.yaw>-720:
							angle.rotation.yaw = 720 + angle.rotation.yaw
						if angle.rotation.yaw>360:
							angle.rotation.yaw = angle.rotation.yaw-360
						if ind>0:
							if self.pre_sel_psi[-1]<90.0 and angle.rotation.yaw>270.0:
								angle.rotation.yaw = angle.rotation.yaw - 360
							elif self.pre_sel_psi[-1]>270.0 and angle.rotation.yaw<90.0:
								angle.rotation.yaw = 360 + angle.rotation.yaw"""
						self.pre_sel_psi.append(angle.rotation.yaw)
						#print(angle.rotation.yaw, end=" ")
						#ind = ind + 1
					print(" ")
					self.pre_sel_psi = np.unwrap(self.pre_sel_psi)
					self.selected_pose = poses
					#print(self.pre_sel_psi)
					cv2.polylines(rgb,np.int32([path_pixels]),False,(0,255,0), 2)
				else:
					cv2.polylines(rgb,np.int32([path_pixels]),False,(0,0,255), 1)
				prev += self.steps[i]
		#cv2.imshow('bv', rgb)
		#cv2.waitKey(10)

	def get_obs_pose_wrt_vehicle(self, ego_pose, obs_pose, p):
		obs_wp = self.m.get_waypoint(obs_pose, project_to_road=True)
		ego_wp = self.m.get_waypoint(ego_pose, project_to_road=True)
		#print('gopwv')
		if obs_wp == None or ego_wp == None:
			print("NONE")
		#print(abs(ego_wp.lane_id), abs(obs_wp.lane_id))
		while abs(ego_wp.lane_id) != abs(obs_wp.lane_id):
			tmp = None
			if abs(ego_wp.lane_id) < abs(obs_wp.lane_id):
				tmp = ego_wp.get_right_lane()
			else:
				tmp = ego_wp.get_left_lane()
			if tmp != None:
				ego_wp = tmp
			else:
				break
			#print(ego_wp, obs_wp)
		#print("abs done ", end="    ")
		min_dist = 3.6
		dist = ego_wp.transform.location.distance(obs_pose)
		#print("dist ", end="    ")
		x = 0
		wp = ego_wp
		#print(wp)
		while dist>min_dist:
			if len(wp.next(2.0)) == 0:
				break
			tmp = wp.next(2.0)[0]
			if tmp == None:
				print("LOL NONE")
			x = x + wp.transform.location.distance(tmp.transform.location)
			dist = tmp.transform.location.distance(obs_pose)
			if x>170:
				break
			wp = tmp
		#print("Gor dist ")
		x = x + wp.transform.location.distance(obs_pose)
		dist = x
		#w1 = self.grp.trace_route(ego_wp.transform.location, obs_pose)#obs_wp.transform.location)
		#dist = 0
		#dist = dist + ego_wp.transform.location.distance(w1[0][0].transform.location)
		#dist = dist + w1[-1][0].transform.location.distance(obs_pose)
		#for i in range(len(w1)-1):
		#    dist = dist + w1[i][0].transform.location.distance(w1[i+1][0].transform.location)
		i = 0
		if p:
			while abs(obs_wp.lane_id) > 1:
				i=1
				break
		else:
			ego_wp = self.m.get_waypoint(ego_pose, project_to_road=True)
			#print(ego_wp.transform.location, ego_pose, obs_pose, end="  ")
			while abs(ego_wp.lane_id) > 1:
				i=1
				break
			#print(ego_wp.transform.location, i)
		y = 1.75 - (3.5*(i))
		#print('gopwv ret')
		return [dist, y]

	def global_to_centerline(self):
		coor = []
		i=0
		ego_pose = self.vehicle.get_transform()
		vel = self.vehicle.get_velocity()
		self.vel = (vel.x**2 + vel.y**2 + vel.z**2)**0.5
		bb = []

		## Ground truth vehicle positions
		"""for obs in self.obs_list:
			pose = []
			lidar = self.sensors[1]
			world_to_lidar = np.asarray(lidar.get_transform().get_inverse_matrix())
			mirror = np.eye(4)

			mirror[1,1] = -1
			
			car_to_world = np.asarray(obs.get_transform().get_matrix())
			box = obs.bounding_box
			extent_x = box.extent.x
			extent_y = box.extent.y
			extent_z = box.extent.z

			vehicle_center = np.array([[box.location.x],[box.location.y],[box.location.z],[1]])

			rect_anchor_x0 = vehicle_center[0,0]-extent_x
			rect_anchor_y0 = vehicle_center[1,0]-extent_y
			rect_anchor_z0 = extent_z

			rect_anchor_x1 = vehicle_center[0,0]-extent_x
			rect_anchor_y1 = vehicle_center[1,0]+extent_y
			rect_anchor_z1 = extent_z

			rect_anchor_x2 = vehicle_center[0,0]+extent_x
			rect_anchor_y2 = vehicle_center[1,0]+extent_y
			rect_anchor_z2 = extent_z

			rect_anchor_x3 = vehicle_center[0,0]+extent_x
			rect_anchor_y3 = vehicle_center[1,0]-extent_y
			rect_anchor_z3 = extent_z

			anchor_0_vec = np.array([[rect_anchor_x0],[rect_anchor_y0],[rect_anchor_z0],[1]])

			anchor_0_vec = mirror@world_to_lidar@car_to_world@anchor_0_vec
			anchor_0_vec = anchor_0_vec[:2,:]

			anchor_1_vec = np.array([[rect_anchor_x1],[rect_anchor_y1],[rect_anchor_z1],[1]])
			anchor_1_vec = mirror@world_to_lidar@car_to_world@anchor_1_vec
			anchor_1_vec = anchor_1_vec[:2,:]

			anchor_2_vec = np.array([[rect_anchor_x2],[rect_anchor_y2],[rect_anchor_z2],[1]])
			anchor_2_vec = mirror@world_to_lidar@car_to_world@anchor_2_vec
			anchor_2_vec = anchor_2_vec[:2,:]

			anchor_3_vec = np.array([[rect_anchor_x3],[rect_anchor_y3],[rect_anchor_z3],[1]])
			anchor_3_vec = mirror@world_to_lidar@car_to_world@anchor_3_vec
			anchor_3_vec = anchor_3_vec[:2,:]


			bb.append([anchor_0_vec, anchor_1_vec, anchor_2_vec, anchor_3_vec])
			
			
			vehicle_center = mirror@world_to_lidar@vehicle_center
			if len(coor)<4:
				pose.append((anchor_0_vec[0][0] + anchor_1_vec[0][0] + anchor_2_vec[0][0] + anchor_3_vec[0][0])/4)
				pose.append((anchor_0_vec[1][0] + anchor_1_vec[1][0] + anchor_2_vec[1][0] + anchor_3_vec[1][0])/4)

				pose.append(obs.get_velocity().x)
				pose.append(obs.get_velocity().y)
				pose.append(((pose[0])**2 + (pose[1])**2)**0.5)
				coor.append(pose)

		coor.append([1100, 0, 0, 0, 1100])
		coor.append([1100, 0, 0, 0, 1100])
		coor.append([1100, 0, 0, 0, 1100])
		coor.append([1100, 0, 0, 0, 1100])"""

		
		## Lidar vehicle positions
		lidar = self.sensor_data[4]
		data = np.frombuffer(lidar.raw_data,  \
				dtype = np.dtype([('x', np.float32), \
				('y', np.float32), ('z', np.float32), \
				('CosAngle', np.float32), ('ObjIdx', np.uint32), \
				('ObjTag', np.uint32)]))

		points = np.array([data['x'], -data['y'], data['z']]).T
		points = points + np.random.normal(loc=0, scale = 0.02, size = (points.shape[0], points.shape[1]))

		labels = np.array(data['ObjTag'])
		agent_ids = np.array(data['ObjIdx'])
		new_data = []
		new_ids = []
		for i in range(labels.shape[0]):
			if( labels[i]==10 and points[i,0]>=0 and agent_ids[i] != 0 ):
				#if (points[i,0]<=0.2 and points[i,0]>=-0.2):
				#	continue 
				new_data.append(points[i,:])
				new_ids.append(agent_ids[i])

		points = np.asarray(new_data)
		agent_ids = np.asarray(new_ids)
		agent_ids_unique = np.unique(agent_ids)
		for i in agent_ids_unique: # Get the bounding box points for all the obstacles
			obs = self.world.get_actor(int(i))
			pose = []
			obs_points = []
			for j in range(points.shape[0]): 
				#print(agent_ids, i, points.shape)
				if(agent_ids[j] == i):
					obs_points.append(points[j,:])
			obs_points = np.asarray(obs_points)
			obs_min_x = min(obs_points[:,0]) - 0.2
			obs_max_x = min(obs_points[:,0]) + 2*2.40
			obs_min_y = min(obs_points[:,1]) - 0.2
			obs_max_y = max(obs_points[:,1]) + 0.2
			pose.append((obs_min_x+obs_max_x)/2)
			pose.append((obs_min_y+obs_max_y)/2)
			pose.append(obs.get_velocity().x)
			pose.append(obs.get_velocity().y)
			pose.append(((pose[0])**2 + (pose[1])**2)**0.5)
			coor.append(pose)
			bb.append([obs_min_x, obs_max_x, obs_min_y, obs_max_y])
		
		while len(coor)<8:
			coor.append([110, 0, 0, 0, 110])
			bb.append([110, 110, 0, 0, 110])
		################

		#ego_wp = self.m.get_waypoint(self.vehicle.get_location(), project_to_road=True)
		#dist = self.prev_pos.distance(self.vehicle.get_location())
		#ego_wp = self.prev_waypoint.next(dist)[0]
		ego_wp = self.m.get_waypoint(self.vehicle.get_location(), project_to_road=True)
		#print(ego_wp.transform.rotation.yaw, end = " ")
		tang_psi = ego_wp.transform.rotation.yaw
		if tang_psi<0 and tang_psi>-360:
			tang_psi = tang_psi+ 360
		elif tang_psi<-360 and tang_psi>-720:
			tang_psi = tang_psi+ 720
		if tang_psi>360:
			tang_psi = tang_psi-360
		orig_psi = ego_pose.rotation.yaw
		if orig_psi<0 and orig_psi>-360:
			orig_psi = orig_psi+ 360
		elif orig_psi<-360 and orig_psi>-720:
			orig_psi = orig_psi+ 720
		if orig_psi>360:
			orig_psi = orig_psi-360
		#print(orig_psi)
		#a = np.unwrap(np.array([tang_psi,orig_psi]))
		#orig_psi = a[1]
		#tang_psi = a[0]
		#print(tang_psi, orig_psi)
		
		if orig_psi<90 and tang_psi>90:
			orig_psi = orig_psi + 360
		elif tang_psi<90 and orig_psi>90:
			orig_psi = orig_psi - 360
		cl_psi = tang_psi - orig_psi
		#while abs(ego_wp.lane_id) > 1:
		#    ego_wp = ego_wp.get_left_lane()
		#print(ego_wp, ego_wp.lane_id)
		if abs(ego_wp.lane_id) != 1:
			ego_wp = ego_wp.get_left_lane()
		#print(ego_wp, ego_wp.lane_id)
		#print(orig_psi, tang_psi)
		#print(self.m.get_waypoint(ego_pose.location, project_to_road=True))
		if abs(self.m.get_waypoint(ego_pose.location, project_to_road=True).lane_id) != 1:
			print(self.m.get_waypoint(ego_pose.location, project_to_road=True).get_left_lane())
		loc = carla.Location()
		loc.x = ego_wp.transform.location.x + 3.5*np.cos(tang_psi*2*np.pi/360 - np.pi/2)
		loc.y = ego_wp.transform.location.y + 3.5*np.sin(tang_psi*2*np.pi/360 - np.pi/2)
		#print(loc, ego_pose.location)
		y = 5.25 - (loc.distance(ego_pose.location))
		self.prev_pos = self.vehicle.get_location()
		self.prev_waypoint = ego_wp
		return [y, cl_psi*2*np.pi/360, vel.x, vel.y], coor, bb

	def centerline_to_global(self, path_x, path_y, path_psi, index):
		#ego_wp = self.m.get_waypoint(self.vehicle.get_location(), project_to_road=True)
		#while abs(ego_wp.lane_id) > 1:
		#    ego_wp = ego_wp.get_left_lane()
		ego_wp = self.prev_waypoint
		path_pose_list = []
		prev_x = 0
		wp = ego_wp
		for i in range(len(path_x)):
			if path_x[i]-prev_x>0.0:
				wp = wp.next(path_x[i]-prev_x)[0]
			wp_tf = wp.transform
			#print(wp_tf, wp.lane_id)
			prev_x = path_x[i]
			y_wp = 1.75 - path_y[i]
			#wp_tf.rotation.yaw = abs(wp_tf.rotation.yaw)
			theta_wp = (wp_tf.rotation.yaw+90)*2*np.pi/360
			wp_tf.location.x = wp_tf.location.x + y_wp*np.cos(theta_wp)
			wp_tf.location.y = wp_tf.location.y + y_wp*np.sin(theta_wp)
			loc = wp_tf.location
			#n_wp = self.m.get_waypoint(loc, project_to_road=False)
			#if n_wp != None:
			#    wp_tf.rotation.yaw = n_wp.transform.rotation.yaw - path_psi[i]*360/(2*np.pi)    
			#else:
			wp_tf.rotation.yaw = wp_tf.rotation.yaw - path_psi[i]*360/(2*np.pi)
			#print(path_x[i], path_y[i], path_psi[i], wp_tf)
			path_pose_list.append(wp_tf)
		return path_pose_list

	def path_to_pixel(self, path_list):
		path_pixels = []
		#ego_wp = self.m.get_waypoint(self.vehicle.get_location(), project_to_road=True)
		vtf = self.vehicle.get_transform()#ego_wp.transform
		#vtf.location.x = self.vehicle.get_transform().location.x
		#vtf.location.y = self.vehicle.get_transform().location.y
		origin_rad = (((self.bv_img_x/2)**2 + (self.bv_img_y/2)**2)**0.5)/4
		origin_theta = np.arctan2(self.bv_img_x/2, self.bv_img_y/2)
		origin_x = origin_rad*np.cos(vtf.rotation.yaw*2*np.pi/360-origin_theta) + vtf.location.x
		origin_y = origin_rad*np.sin(vtf.rotation.yaw*2*np.pi/360-origin_theta) + vtf.location.y
		for i in path_list:
			pixel_rad = (((origin_x-i.location.x)**2 + (origin_y-i.location.y)**2)**0.5)*4
			pixel_theta = np.arctan2(origin_y-i.location.y, origin_x-i.location.x)
			if pixel_theta<0:
				pixel_theta = 2*np.pi + pixel_theta
			pixel_theta = pixel_theta*360/(2*np.pi)
			rgb_theta = vtf.rotation.yaw-pixel_theta
			rgb_x = pixel_rad*np.sin(rgb_theta*2*np.pi/360)
			rgb_y = pixel_rad*np.cos(rgb_theta*2*np.pi/360)
			path_pixels.append([rgb_x, rgb_y])
		return path_pixels
	
	def destroy(self):
		for actor in self.actor_list:
			actor.destroy()

	def simulate(self):
		if self.should_quit():
			return
		#print("Ticking world.....", end='')
		self.clock.tick()
		self.tick(timeout=2.0)
		#print("Done")

	def apply_control(self, v, w, target_acc, prev_acc):
		physics_control = self.vehicle.get_physics_control()
		max_steer_angle_list = []
		# For each Wheel Physics Control, print maximum steer angle
		for wheel in physics_control.wheels:
			max_steer_angle_list.append(wheel.max_steer_angle)
		max_steer_angle = max(max_steer_angle_list)*np.pi/180

		throttle_lower_border = -(0.01*9.81*physics_control.mass + 0.5*0.3*2.37*1.184*self.vel**2 + \
			9.81*physics_control.mass*np.sin(self.vehicle.get_transform().rotation.pitch*2*np.pi/360))/physics_control.mass

		brake_upper_border = throttle_lower_border + -500/physics_control.mass
		self.pid.setpoint = target_acc
		acc = (self.vel - self.prev_vel)/0.08
		if acc>10:
			control = self.pid(0)
		else:
			self.prev_acc = (self.prev_acc*4 + acc)/5
			#acc = self.vehicle.get_acceleration()
			#acc = (acc.x**2 + acc.y**2 + acc.z**2)**0.5
			control = self.pid(self.prev_acc)
		steer = np.arctan(w*3.2/v)
		steer = -steer/max_steer_angle
		throttle = 0
		brake = 0
		self.throttle = np.clip(self.throttle + control,-4.0, 4.0)
		if self.throttle>throttle_lower_border:
			throttle = (self.throttle-throttle_lower_border)/4
			brake = 0
		elif self.throttle> brake_upper_border:
			brake = 0
			throttle = 0
		else:
			brake = (brake_upper_border-self.throttle)/4
			throttle = 0
		brake = np.clip(brake, 0.0, 1.0)
		throttle = np.clip(throttle, 0.0, 1.0)
		self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
		#yaw = self.vehicle.get_transform().rotation.yaw*2*np.pi/360
		#self.vehicle.set_target_velocity(carla.Vector3D(x=v*np.cos(yaw), y=v*np.sin(yaw)))
		#self.vehicle.set_target_angular_velocity(carla.Vector3D(z=-w*360/(2*np.pi)))
		self.prev_vel = self.vel
		#print("Throttle = ", self.throttle, "Steer = ", steer, "Throttle = ", throttle, "Acc = ", self.prev_acc)

	def move_vehicles(self):
		vel_list = [3.0, 4.0, 5.0]
		for obs in self.obs_list:
			vel = random.choice(vel_list)
			#obs.set_target_velocity(carla.Vector3D(vel,0.0,0))
			self.tm.auto_lane_change(obs,False)
			#obs.set_autopilot(True)
		#self.vehicle.set_target_velocity(carla.Vector3D(7.5,0.0,0))
		self.vehicle.set_target_velocity(carla.Vector3D(7.5,0.0,0))

	def teleport_obstacles(self, obs_list):
		for i in range(len(obs_list)):
			if obs_list[i][0]<-30:
				obs = self.obs_list[i]
				obs_wp = self.m.get_waypoint(obs.get_location(), project_to_road=True)
				print(obs_wp.next(140))
				obs_wp = obs_wp.next(175)[0]
				obs.set_transform(obs_wp.transform)


		"""for obs in self.behind_ego:
			if obs == None:
				continue
			obs_wp = self.m.get_waypoint(obs.get_location(), project_to_road=True)
			ego_wp = self.m.get_waypoint(self.vehicle.get_location(), project_to_road=True)
			while abs(ego_wp.lane_id) != abs(obs_wp.lane_id):
				if abs(ego_wp.lane_id) < abs(obs_wp.lane_id):
					ego_wp = ego_wp.get_right_lane()
				else:
					ego_wp = ego_wp.get_left_lane()
			w1 = self.grp.trace_route(obs_wp.transform.location, ego_wp.transform.location)
			dist = 0
			dist = dist + obs_wp.transform.location.distance(w1[0][0].transform.location)
			for i in range(len(w1)-1):
				dist = dist + w1[i][0].transform.location.distance(w1[i+1][0].transform.location)
			dist = dist + w1[-1][0].transform.location.distance(ego_wp.transform.location)
			if dist>36:
				print("check1")
				obs_wp = ego_wp.next(140)[0]
				obs.set_transform(obs_wp.transform)"""
	

	def projection_new(self, num=20):
		meters_per_frame = 1.0
		number_of_lanepoints = num
		waypoint_list = []
		waypoint = self.m.get_waypoint(self.vehicle.get_location(), project_to_road=True)
		waypoint_list.append(waypoint)
		is_wp_junction = False
		is_junction_max = int(5/meters_per_frame)
		is_junction_ind = 0
		#print(waypoint)
		for i in range(0, number_of_lanepoints):
			next_waypoint = random.choice(waypoint_list[-1].next(meters_per_frame))

			while next_waypoint.get_right_lane() and next_waypoint.get_right_lane().lane_type == carla.LaneType.Driving:
				next_waypoint = next_waypoint.get_right_lane()
			#print(next_waypoint)
			waypoint_list.append(next_waypoint)
		
		orientationVec = waypoint_list[1].transform.get_forward_vector()
		
		length = math.sqrt(orientationVec.y*orientationVec.y+orientationVec.x*orientationVec.x)
		abVec = carla.Location(orientationVec.y,-orientationVec.x,0) / length * 0.5* waypoint_list[1].lane_width
		left_lanemarking = waypoint_list[1].transform.location + abVec
		lpt = [[],[],[]]
		#print("Start pose", waypoint_list[0], waypoint_list[0].lane_id)
		for i in range(1, len(waypoint_list)):
			orientationVec = waypoint_list[i].transform.get_forward_vector()
			length = math.sqrt(orientationVec.y*orientationVec.y+orientationVec.x*orientationVec.x)
			abVec = carla.Location(orientationVec.y,-orientationVec.x,0) / length * 0.5* waypoint_list[i].lane_width
			right_lanemarking = waypoint_list[i].transform.location - abVec
			#print(waypoint_list[i].right_lane_marking.type, waypoint_list[i], waypoint_list[i].lane_type) 
			lpt[0].append(right_lanemarking) if(waypoint_list[i].right_lane_marking.type != carla.LaneMarkingType.NONE) else lpt[0].append(None)
			#Town 05lpt[0].append(right_lanemarking) if(not waypoint_list[i].is_junction) else lpt[0].append(None)
			#if lpt[0][-1] != None:
			#    print("index = ", i, waypoint_list[i], waypoint_list[i].lane_id, 0, lpt[0][-1].x, lpt[0][-1].y)
			#lpt[1].append(left_lanemarking)
			idx = 1
			l_id = waypoint_list[i].lane_id
			while waypoint_list[i].lane_id*l_id>0:
				#print(idx, waypoint_list[i].lane_id, waypoint_list[i], l_id)
				#if waypoint_list[i].is_junction:
				#    lpt[idx].append(None)
				#    waypoint_list[i] = waypoint_list[i].get_left_lane()
				if waypoint_list[i] and waypoint_list[i].left_lane_marking.type != carla.LaneMarkingType.NONE:
					lpt[idx].append(waypoint_list[i].transform.location + abVec)
					#print(waypoint_list[i], waypoint_list[i].lane_id, idx, lpt[idx][-1].x, lpt[idx][-1].y)
					waypoint_list[i] = waypoint_list[i].get_left_lane()
				else:
					lpt[idx].append(None)
					break
				if waypoint_list[i] == None:
					break
				#print(idx, waypoint_list[i].lane_id)
				idx+=1
			#print("\n")

		lane_lists = lpt
		#print(lane_lists)
		lanes_points = []
		world_coords = []
		##############################################################
		##############################################################
		for lane_list in lane_lists:
			flat_lane_list_a = []
			flat_lane_list_b = []
			lane_list = list(filter(lambda x: x!= None, lane_list))
			camera_rgb = self.sensors[0]
			fl = 800/(2* np.tan(90 * np.pi / 360))
			cameraMatrix = np.array([[fl, 0.0, 800/2],
							[0.0, fl, 600/2],
							[0.0, 0.0, 1]])
			
			if lane_list:    
				last_lanepoint = lane_list[0]
				
			for lanepoint in lane_list:
				if(lanepoint and last_lanepoint):
					# Draw outer lanes not on junction
					distance = math.sqrt(math.pow(lanepoint.x-last_lanepoint.x ,2)+math.pow(lanepoint.y-last_lanepoint.y ,2)+math.pow(lanepoint.z-last_lanepoint.z ,2))
				
					# Check of there's a hole in the list
					if distance > meters_per_frame * 3:
						#print("hole")
						flat_lane_list_b = flat_lane_list_a
						flat_lane_list_a = []
						last_lanepoint = lanepoint
						continue
					
					last_lanepoint = lanepoint
					flat_lane_list_a.append([lanepoint.x, lanepoint.y, lanepoint.z, 1.0])
					#print(flat_lane_list_a[-1], len(flat_lane_list_a))

				else:
					# Just append a "Null" value
					flat_lane_list_a.append([None, None, None, None])
			#print("fla = ", len(flat_lane_list_a))
			if flat_lane_list_a:
				world_points = np.float32(flat_lane_list_a).T
				
				# This (4, 4) matrix transforms the points from world to sensor coordinates.
				world_2_camera = np.array(camera_rgb.get_transform().get_inverse_matrix())
				
				# Transform the points from world space to camera space.
				sensor_points = np.dot(world_2_camera, world_points)
				#print(sensor_points)
				#quit()
				world_coords.append(sensor_points)
				
				# Now we must change from UE4's coordinate system to a "standard" one
				# (x, y ,z) -> (y, -z, x)
				point_in_camera_coords = np.array([sensor_points[1],
												sensor_points[2] * -1,
												sensor_points[0]])
				
				# Finally we can use our intrinsic matrix to do the actual 3D -> 2D.
				points_2d = np.dot(cameraMatrix, point_in_camera_coords)
				
				# Remember to normalize the x, y values by the 3rd value.
				points_2d = np.array([points_2d[0, :] / points_2d[2, :],
									points_2d[1, :] / points_2d[2, :],
									points_2d[2, :]])
				
				# visualize everything on a screen, the points that are out of the screen
				# must be discarted, the same with points behind the camera projection plane.
				points_2d = points_2d.T
				points_in_canvas_mask = (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < 800) & \
										(points_2d[:, 1] > 0.0) & (points_2d[:, 1] < 600) & \
										(points_2d[:, 2] > 0.0)
				
				points_2d = points_2d[points_in_canvas_mask]
				
				# Extract the screen coords (xy) as integers.
				x_coord = points_2d[:, 0].astype(np.int)
				y_coord = points_2d[:, 1].astype(np.int)
			else:
				x_coord = []
				y_coord = []

			if flat_lane_list_b:
				world_points = np.float32(flat_lane_list_b).T
				world_coords.append(world_points)
				
				# This (4, 4) matrix transforms the points from world to sensor coordinates.
				world_2_camera = np.array(camera_rgb.get_transform().get_inverse_matrix())
				
				# Transform the points from world space to camera space.
				sensor_points = np.dot(world_2_camera, world_points)
				#print(sensor_points)
				#quit()
				world_coords.append(sensor_points)
				
				# Now we must change from UE4's coordinate system to a "standard" one
				# (x, y ,z) -> (y, -z, x)
				point_in_camera_coords = np.array([sensor_points[1],
												sensor_points[2] * -1,
												sensor_points[0]])
				
				# Finally we can use our intrinsic matrix to do the actual 3D -> 2D.
				points_2d = np.dot(self.cameraMatrix, point_in_camera_coords)
				
				# Remember to normalize the x, y values by the 3rd value.
				points_2d = np.array([points_2d[0, :] / points_2d[2, :],
									points_2d[1, :] / points_2d[2, :],
									points_2d[2, :]])
				
				# visualize everything on a screen, the points that are out of the screen
				# must be discarted, the same with points behind the camera projection plane.
				points_2d = points_2d.T
				points_in_canvas_mask = (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < 800) & \
										(points_2d[:, 1] > 0.0) & (points_2d[:, 1] < 600) & \
										(points_2d[:, 2] > 0.0)
				
				points_2d = points_2d[points_in_canvas_mask]
				
				old_x_coord = np.insert(x_coord, 0, -1)
				old_y_coord = np.insert(y_coord, 0, -1)
				
				# Extract the screen coords (xy) as integers.
				new_x_coord = points_2d[:, 0].astype(np.int)
				new_y_coord = points_2d[:, 1].astype(np.int)

				x_coord = np.concatenate((new_x_coord, old_x_coord), axis=None)
				y_coord = np.concatenate((new_y_coord, old_y_coord), axis=None)
			#print(x_coord, y_coord)
			lane_list = list(zip(x_coord, y_coord))

			###########################################################
			###########################################################

			image_seg = self.sensor_data[3]
			array = np.frombuffer(image_seg.raw_data, dtype=np.dtype("uint8"))
			array = np.reshape(array, (image_seg.height, image_seg.width, 4))
			image = array[:, :, :3]
			filtered_lane_list = []
			for lanepoint in lane_list:
				x = int(lanepoint[0])
				y = int(lanepoint[1])
				if(np.any(image[y][x] == (128, 64, 128), axis=-1) or   # Road
				np.any(image[y][x] == (157, 234, 50), axis=-1) or   # Roadline
				np.any(image[y][x] == (244, 35, 232), axis=-1) or   # Sidewalk
				np.any(image[y][x] == (220, 220, 0), axis=-1) or    # Traffic sign
				np.any(image[y][x] == (0, 0, 142), axis=-1)):       # Vehicle
					filtered_lane_list.append(lanepoint)
				else:
					filtered_lane_list.append((-2, y))
			lanes_points.append(filtered_lane_list)
				
		return lanes_points, world_coords


def get_font():
		fonts = [x for x in pygame.font.get_fonts()]
		default_font = 'ubuntumono'
		font = default_font if default_font in fonts else fonts[0]
		font = pygame.font.match_font(font)
		return pygame.font.Font(font, 14)

def main():
	pygame.init()
	display = pygame.display.set_mode(
			(800, 600),
			pygame.HWSURFACE | pygame.DOUBLEBUF)
	font = get_font()
	clock = pygame.time.Clock()
	csm = CarlaSyncBatch(display, font, clock, n_obs=12)
	path_x = np.linspace(0, 120, num=100)
	path_y = np.linspace(5.25, -5.25, num=100)
	ang = np.arctan2(-10.25, 120)
	path_psi = np.linspace(ang, ang, num=100)
	try:
		print("Starting........")
		while True:
			if csm.should_quit():
				break
			clock.tick()
			csm.tick(timeout=2.0)
			#ego_pose, obs_poses = csm.global_to_centerline()
			poses = csm.centerline_to_global(path_x, path_y, path_psi)
			csm.path_to_pixel(poses)
			csm.visualize()
			print(clock.get_fps())
			print('-------------------------------------------')
			#break
			
	finally:

		print('destroying actors.')
		csm.destroy()

		pygame.quit()
		print('done.')
	
#main()
