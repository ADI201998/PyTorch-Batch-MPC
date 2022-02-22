
from math import radians
import random
import numpy as np
import pygame
import time
import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader
import os
from pathlib import Path
import cv2

from scipy import spatial
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy import interpolate
from scipy.interpolate import PPoly
from sympy import *
from collections import deque


dir_name = os.path.abspath('')


from matplotlib.patches import Ellipse, Rectangle
from shapely.geometry import Point
from shapely.affinity import scale, rotate


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from msgs_car.msg import States, Controls
import lstr.batch_2_lane as cbs

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import pprint
import argparse
import importlib
import sys

pkg_path = str(Path(dir_name+str("/src/LSTR")))
try:
    print(pkg_path)
    sys.path.append(pkg_path)
except:
    print("NOO")
    pass

from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
from db.utils.evaluator import Evaluator

from torch.autograd import Variable
import torch.nn.functional as F

from torch import nn

from utils import crop_image, normalize_

torch.backends.cudnn.benchmark = False

# from highway_car import ngsim_data


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_curves = outputs['pred_logits'], outputs['pred_curves']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        labels[labels != 1] = 0
        results = torch.cat([labels.unsqueeze(-1).float(), out_curves], dim=-1)

        return results



class Lane_optimizer_exp():
    
    def __init__(self):
        
        # paramsL: will be used later to clean data if required ( remove points which have z not lying on the ground plane ) 
        self.centerline_params = []
        self.lane_width = 3.5
    
    def update_vals(self, X, Y, limits, scale, bounds):
        self.X_input = np.array(X)
        self.Y_input = np.array(Y)
        self.scale = scale
        self.shape = np.cumsum(limits)
        self.limits = np.zeros((len(limits)))
        self.bounds = bounds
        for i in range(len(limits)):
            self.limits[i] = self.shape[i]
        self.params = self.fit_curve()
        if len(self.limits)%2 == 0:
            self.centerline_params = [self.params[0], self.params[1], self.scale + self.params[2] + 3.5*(len(self.limits)/2-0.5) - self.params[0]]
        else:
            self.centerline_params = [self.params[0], self.params[1], self.scale + self.params[2] + 3.5*(len(self.limits)/2-0.5) - self.params[0]]

    def fit_curve(self):
        popt,pcov = curve_fit(self.func_exp,self.X_input,self.Y_input, maxfev=5000)
        return popt

    def get_eqns(self):

        ''' returns the equations to the three lane lines '''
        params = []
        for i in range(len(self.limits)):
            params.append([self.params[0], self.params[1], self.scale + self.params[2]+3.5*i] - self.params[0])
        return params

    def func_exp(self,X,a,b,c):
        Y = []
        ind = 0
        for i in range(len(self.limits)):
            x = X[ind:int(self.limits[i])]
            Y.append(a * np.exp(b*x) + c+3.5*i -a)
            ind = int(self.limits[i])
        y = [i for l in Y for i in l]
    
        return np.array(y)



class Lane_optimizer_quadratic():
    
    def __init__(self):
        
        # paramsL: will be used later to clean data if required ( remove points which have z not lying on the ground plane ) 
        self.centerline_params = []
        self.params_old = [0, 0]
    
    def update_vals(self, X, Y, limits, scale, bounds):
        self.X_input = np.array(X)
        self.Y_input = np.array(Y)
        self.scale = scale
        self.shape = np.cumsum(limits)
        self.limits = np.zeros((len(limits)))
        self.bounds = bounds
        for i in range(len(limits)):
            self.limits[i] = self.shape[i]
        self.params = self.fit_curve()
        self.params[:2] = 0.8*np.array(self.params_old) + 0.2*np.array([self.params[0], self.params[1]])
        self.params_old = self.params[:2]
        if len(self.limits)%2 == 0:
            self.centerline_params = [self.params[0], self.params[1], self.scale + self.params[2]*(len(self.limits)/2-0.5)]
        else:
            self.centerline_params = [self.params[0], self.params[1], self.scale + self.params[2]*(len(self.limits)/2-0.5)]

    def fit_curve(self):
        popt,pcov = curve_fit(self.func_2d,self.X_input,self.Y_input,bounds=self.bounds, method='dogbox')
        return popt

    def get_eqns(self):

        ''' returns the equations to the three lane lines '''
        params = []
        for i in range(len(self.limits)):
            params.append([self.params[0], self.params[1], self.scale + self.params[2]*i])
        return params

    def func_2d(self,X,a,b,c):
        Y = []
        ind = 0
        for i in range(len(self.limits)):
            x = X[ind:int(self.limits[i])]
            Y.append(a * np.power(x,2) + b * x + c*i)
            ind = int(self.limits[i])
        y = [i for l in Y for i in l]
    
        return np.array(y)



class Lane_optimizer_cubic():
    
    def __init__(self):
        
        # paramsL: will be used later to clean data if required ( remove points which have z not lying on the ground plane ) 
        self.centerline_params = []
    
    def update_vals(self, X, Y, limits, scale, bounds):
        self.X_input = np.array(X)
        self.Y_input = np.array(Y)
        self.scale = scale
        self.shape = np.cumsum(limits)
        self.limits = np.zeros((len(limits)))
        self.bounds = bounds
        for i in range(len(limits)):
            self.limits[i] = self.shape[i]
        self.params = self.fit_curve()
        if len(self.limits)%2 == 0:
            self.centerline_params = [self.params[0], self.params[1], self.params[2], self.scale + self.params[3]*(len(self.limits)/2-0.5)]
        else:
            self.centerline_params = [self.params[0], self.params[1], self.params[2], self.scale + self.params[3]*(len(self.limits)/2-0.5)]

    def fit_curve(self):
        popt,pcov = curve_fit(self.func_2d,self.X_input,self.Y_input,bounds=self.bounds, method='dogbox')
        return popt

    def get_eqns(self):

        ''' returns the equations to the three lane lines '''
        params = []
        for i in range(len(self.limits)):
            params.append([self.params[0], self.params[1], self.params[2], self.scale + self.params[3]*i])
        return params

    def get_p_dist(self, xp, yp):
        x = Symbol('x')
        D_sq = (x-xp)**2 + (self.centerline_params[0]*x**3 + self.centerline_params[1]*x**2 + self.centerline_params[2]*x + self.centerline_params[3] - yp)**2
        D_sq_dot = D_sq.diff(x)
        x_0 = solve(Eq(D_sq_dot, 0), x)[0]
        min_dist = D_sq.subs({x:x_0})
        return min_dist**0.5

    def func_2d(self,X,a,b,c,d):
        Y = []
        ind = 0
        for i in range(len(self.limits)):
            x = X[ind:int(self.limits[i])]
            Y.append(a * np.power(x,3) + b * np.power(x,2) + c * x + d*i)
            ind = int(self.limits[i])
        y = [i for l in Y for i in l]
    
        return np.array(y)


class MinimalSubscriber(Node):

    def __init__(self, display, font, clock):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Controls,
            'ego_vehicle_cmds',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.plot_ellipse = 0


        self.publisher_ = self.create_publisher(States, 'ego_vehicle_obs', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        random.seed(0)
        self.a_ell = 4.9
        self.b_ell = 2.625
        self.a_rect = 5.0
        self.b_rect = 2.5
        self.steps = []
        self.omega = 0.0
        self.num_lanes = 0

        self.cnt = 1
        self.loop = -1
        self.index = 0
        self.intersection = [False, False, False, False, False, False]
        self.upper = 120 #130 for cruisie
        self.lower_lim = -30
        self.upper_lim = self.upper
        self.pre_x = np.array([])
        self.pre_y = np.array([])
        self.pre_psi = np.array([])
        self.Gotit = 1.0
        self.v_controls = np.array([])
        self.psi_constrols = np.array([])
        self.num_goal = 1
        #self.L = Lane_optimizer_cubic()
        self.L = Lane_optimizer_quadratic()
        #self.L = Lane_optimizer_exp()
        self.x_prev = deque(maxlen=10)
        self.mv_psi = 0.0

        with open('src/highway_car/config.yaml') as f:
            data = yaml.load(f, Loader=SafeLoader)
            setting = str(data["setting"])
        # cruise = 0, Right Lane = 1, High Speed RightLane = 2, NGSIM = 3
        if setting == "cruise_IDM":
            self.setting = 0 

        elif setting == "RL_IDM":
            self.setting = 1
        elif setting == "HSRL_IDM":
            self.setting = 2
        else:
            self.setting = 3

        if self.setting == 3:
            self.NGSIM = True
        else:
            self.NGSIM = False
        print (self.setting)
        self.num_obs = 6
        self.obs = np.zeros([self.num_obs+1, 4])

        # if self.NGSIM == False:
        #     self.obs[0] = [0, -10, 14.0, 0.0]
        # else:
        #     self.obs[0] =[-10, -2, 14.0, 0.0]

        self.obs[0] = [0, 10, 7.5, 0.0]
        #self.obs = np.array([])

                
            
        self.ours_x = []
        self.ours_y = []



        self.v = self.obs[0][2]
        self.prev_vel = self.v
        self.prev_acc = 0.0
        self.w = 0.0 

        self.prev_psi = 0.0
        self.psi = 0.0
        self.psi_act = 0.0
        self.dt = 0.08
    
        self.sim_time = np.array([])
        self.min_x = 0

        self.flag = 1
        self.fig = plt.figure(0)
        self.ax1 = self.fig.add_subplot(221, aspect='equal')
        self.ax2 = self.fig.add_subplot(223, aspect='equal')
        self.ax3 = self.fig.add_subplot(222, aspect='equal')
        mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        self.fig.set_size_inches(15, 20)

        test_iter = 400000
        cfg_file = os.path.join(system_configs.config_dir, "LSTR" + ".json")
        print("cfg_file: {}".format(cfg_file))
        with open(cfg_file, "r") as f:
                configs = json.load(f)
        configs["system"]["snapshot_name"] = 'LSTR'
        system_configs.update_config(configs["system"])

        print("building neural network...")
        self.nnet = NetworkFactory()
        print("Done")
        image_root = "/home/aditya/Documents/LSTR"

        fl = 800/(2* np.tan(90 * np.pi / 360))
        self.c_int = np.array([[fl, 0.0, 800/2],
                        [0.0, fl, 600/2],
                        [0.0, 0.0, 1]])

        self.c_int_inv = np.linalg.inv(self.c_int)
        self.l2ct = np.array([[1, 0.0, 0.0, 2.0],
                        [0.0, 1, 0.0, 0],
                        [0.0, 0.0, 1, -0.4],
                        [0.0, 0.0, 0.0, 1]])
        self.c2lt = np.array([[1, 0.0, 0.0, -2.0],
                        [0.0, 1, 0.0, -0],
                        [0.0, 0.0, 1, 0.4],
                        [0.0, 0.0, 0.0, 1]])

        #nnet = torch.load(image_root+"/nnnet")
        self.nnet.load_params(test_iter)
        self.nnet.cuda()

        self.csm = cbs.CarlaSyncBatch(display, font, clock, n_obs=8)
        self.csm.vis_path = False
        self.csm.num_goal = self.num_goal
        self.csm.simulate()
        self.csm.move_vehicles()
        self.other_vehicles = [[-1000, -1000, 0,0,1000],
                                [-1010, -1010, 0,0,1000],
                                [-1020, -1020, 0,0,1000],
                                [-1030, -1030, 0,0,1000],
                                [-1040, -1040, 0,0,1000],
                                [-1050, -1050, 0,0,1000]]
        self.other_vehicles_cl = [[-10,-10,-10, -10],
                                    [-10,-10,-10, -10],
                                    [-10,-10,-10, -10],
                                    [-10,-10,-10, -10],
                                    [-10,-10,-10, -10],
                                    [-10,-10,-10, -10]]

        
        print("STARTING SIMULATION")


    def func_2d(self, X,a,b,c):
        return a * np.power(X,2) + b * X + c

    def func_3d(self, X,a,b,c,d):
        return a * np.power(X,3) + b * np.power(X,2) + c * X + d

    def integrand_3d(self, X, a, b, c):
        return (1 + (3*a*np.power(X-self.min_x,2) + 2*b*(X-self.min_x) + c)**2)**0.5
    
    def integrand_2d(self, X, a, b):
        return (1 + (2*a*(X-self.min_x) + b)**2)**0.5
    
    def create_ellipse(self, center, axes, inclination):
        p = Point(*center)
        c = p.buffer(1)
        ellipse = scale(c, *axes)
        ellipse = rotate(ellipse, inclination)
        return ellipse

    def checkCollision(self):
        
        obs_ellipse = [self.create_ellipse((self.obs[i][0], self.obs[i][1]), (self.a_ell/2, self.b_ell/2), 0) for i in range(self.num_obs)]
        ego_ellipse = self.create_ellipse((self.obs[0][0], self.obs[0][1]), (self.a_ell/2, self.b_ell/2), 0*self.psi*180.0/np.pi)

        self.intersection = [ego_ellipse.intersects(obs_ellipse[i]) for i in range(self.num_obs)]
        for i in range(self.num_obs):
            if self.intersection[i]:
                ptsx, ptsy = ego_ellipse.intersection(obs_ellipse[i]).exterior.coords.xy
                if len(ptsx) < 10:
                    self.intersection[i] = False
        self.intersection = any([inter == True for inter in self.intersection])


    def listener_callback(self, msg):
        
        self.Gotit = 1.0
        self.v = msg.v
        self.w = msg.w
        #print(self.v, self.w)
        
        xx  = 0
        cnt = 0
        self.steps = []
        self.pre_x = np.array([])
        self.pre_y = np.array([])
        self.pre_psi = np.array([])
        for i in msg.batch.poses:
            self.pre_x = np.append(self.pre_x, i.position.x)
            self.pre_y = np.append(self.pre_y, i.position.y)
            self.pre_psi = np.append(self.pre_psi, i.orientation.z)
            #if abs(i.position.x - self.pre_x[0]) < 0.001 and len(self.pre_x) > 1:
            #    self.steps.append(cnt)
            #    cnt = 0
            cnt+=1
            if cnt == 100:
                self.steps.append(cnt)
                cnt = 0
        self.steps.append(cnt)
        print(self.pre_psi)
        self.num_goal = msg.goals
        self.csm.num_goal = self.num_goal
        self.index = msg.index
        self.v_controls = np.append(self.v_controls, self.v)

    def lane_pred(self, nnet, img):
        input_size  = [360, 640]

        postprocessors = {'bbox': PostProcess()}
        image         = img
        height, width = image.shape[0:2]

        images = np.zeros((1, 3, input_size[0], input_size[1]), dtype=np.float32)
        masks = np.ones((1, 1, input_size[0], input_size[1]), dtype=np.float32)
        orig_target_sizes = torch.tensor(input_size).unsqueeze(0).cuda()
        pad_image     = image.copy()
        pad_mask      = np.zeros((height, width, 1), dtype=np.float32)
        resized_image = cv2.resize(pad_image, (input_size[1], input_size[0]))
        resized_mask  = cv2.resize(pad_mask, (input_size[1], input_size[0]))
        masks[0][0]   = resized_mask.squeeze()
        #cv2.imshow("img", resized_image)
        #cv2.waitKey()
        resized_image = resized_image / 255.
        mean = [0.40789655, 0.44719303, 0.47026116]
        std = [0.2886383,  0.27408165, 0.27809834]
        #mean = [np.mean(resized_image), np.mean(resized_image), np.mean(resized_image)]
        #std = [np.std(resized_image),  np.std(resized_image), np.std(resized_image)]
        #print(mean, std)
        normalize_(resized_image, mean, std)
        resized_image = resized_image.transpose(2, 0, 1)
        images[0]     = resized_image
        images        = torch.from_numpy(images).cuda(non_blocking=True)
        masks         = torch.from_numpy(masks).cuda(non_blocking=True)
        torch.cuda.synchronize(0)  # 0 is the GPU id
        #t0            = time.time()
        outputs, weights = nnet.test([images, masks])
        torch.cuda.synchronize(0)  # 0 is the GPU id
        #t = time.time() - t0
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        pred = results[0].cpu().numpy()
        img  = pad_image
        img_h, img_w, _ = img.shape
        pred = pred[pred[:, 0].astype(int) == 1]
        overlay = img.copy()
        color = (0, 255, 0)
        pred_points = []
        for i, lane in enumerate(pred):
            lane = lane[1:]  # remove conf
            lower, upper = lane[0], lane[1]
            lane = lane[2:]  # remove upper, lower positions

            # generate points from the polynomial
            ys = np.linspace(lower, upper, num=100)
            points = np.zeros((len(ys), 2), dtype=np.int32)
            points[:, 1] = (ys * img_h).astype(int)
            points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys -
                                lane[5]) * img_w).astype(int)
            points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]
            pred_points.append(points)
            # draw lane with a polyline on the overlay
            for current_point, next_point in zip(points[:-1], points[1:]):
                overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=15)

            # draw lane ID
            if len(points) > 0:
                cv2.putText(img, str(i), tuple(points[0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=color,
                            thickness=3)
        # Add lanes overlay
        w = 0.6
        img = ((1. - w) * img + w * overlay).astype(np.uint8)

        cv2.imshow('img', img)
        self.ax3.imshow(img)
        #cv2.waitKey(5)
        #print(img.shape)
        return pred_points


    def projection_cubic(self):
        """image_rgb = self.csm.sensor_data[1]
        array = np.frombuffer(image_rgb.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image_rgb.height, image_rgb.width, 4))
        array = array[:, :, :3]
        lane_pt, world_pt = self.csm.projection_new(num=30)
        img = array.copy()
        for points in lane_pt:
            for p in points:
                if p[0]>=0 and p[1]>=0:
                    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)
        cv2.imshow("points", img)
        cv2.waitKey(1)

        X_input = np.array(world_pt)
        limits = []
        X = np.array([])
        Y = np.array([])
        min_ys = []
        for i in X_input:
            min_ys.append(-i[1][0])
        
        min_y = np.sort(min_ys)[0]
        ind_min = np.argsort(min_ys)
        for l in ind_min:
            #print(l,min_ys, min_y)
            #print(X_input[l])
            limits.append(len(X_input[l][0]))
            X = np.concatenate((X, X_input[l][0]))
            Y = np.concatenate((Y, -1*X_input[l][1]))
        s2 = time.time()
        #print(s2-s1)"""

        image_rgb = self.csm.sensor_data[1]
        point_cloud = self.csm.sensor_data[2]
        array = np.frombuffer(image_rgb.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image_rgb.height, image_rgb.width, 4))
        array = array[:, :, :3]
        #pcd = o3d.geometry.PointCloud()

        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        num_channel_pts = []
        num_channel_pts.append(0)
        s1 = time.time()
        for i in range(point_cloud.channels):
            num_channel_pts.append(point_cloud.get_point_count(i)+num_channel_pts[-1])
        points = data[:, :-1]
        #points[:, 1] = -points[:, 1]
        channelwise_pts = []
        for i in range(64):
            channel_pt_ap = points[num_channel_pts[i]:num_channel_pts[i+1],:]
            channelwise_pts.append(channel_pt_ap)
        #channelwise_pts = [i[np.where((np.arctan2(i[:,1], i[:,0])<np.pi/4) & (np.arctan2(i[:,1], i[:,0])>-np.pi/4))[0]] for i in channelwise_pts]
        channelwise_pts = [ (self.c2lt@np.vstack((i.T, np.ones((1, i.T.shape[1])))))[:3,:].T for i in channelwise_pts]
        channelwise_pts = [ i[np.where((i[:,0]>0.0))[0]] for i in channelwise_pts]
        channelwise_pixels = [ self.c_int@i[:, [1,2,0]].T for i in channelwise_pts]
        channelwise_pixels = [ (i/i[2:,].reshape(1, i.shape[1])).T[:, :2] for i in channelwise_pixels]
        channelwise_indices = [ np.where((i[:,0]<800) & (i[:,0]>0) & (i[:,1]<600) & (i[:,1]>0))[0] for i in channelwise_pixels]
        channelwise_projection_data = [[channelwise_pts[i][channelwise_indices[i]], channelwise_pixels[i][channelwise_indices[i]]] for i in range(len(channelwise_indices))]
        idx = 0
        min_indices = []
        all_lane_pts = []
        all_lane_pix_pts = []
        s2= time.time()
        #print(s2-s1)
        lid_img2 = array.copy()
        s1 = time.time()
        all_lane_pix = self.lane_pred(self.nnet, array)
        s2 = time.time()
        #print(s2-s1)

        s1 = time.time()
        for pix_pts in channelwise_projection_data:
            pix_pts[1][:, 1] = 600 - pix_pts[1][:, 1]
        lane_pts = []
        lane_pix_pts = []
        if len(all_lane_pix)>0:
            for lane in all_lane_pix:
                lp_pt = []
                lp_pix_pt = []
                for pts in channelwise_projection_data:
                    if pts[1].shape[0] == 0 or pts[1].shape[1] == 0:
                        continue
                    if np.max(pts[1][:,1]) < np.max(np.array(lane)[:, 1]) and np.min(pts[1][:,1]) > np.min(np.array(lane)[:, 1]):
                        dists = spatial.distance.cdist(pts[1], np.array(lane))
                        min_id = np.argwhere(dists == np.min(dists))[0][0]
                        lp_pt.append(pts[0][min_id])
                        lp_pix_pt.append(pts[1][min_id])
                        all_lane_pts.append(pts[0][min_id])
                        all_lane_pix_pts.append(pts[1][min_id])
                lane_pts.append(lp_pt)
                lane_pix_pts.append(lp_pix_pt)   

        for p in all_lane_pix_pts:
            cv2.circle(lid_img2, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)
        cv2.imshow("lanes projection", lid_img2)
        cv2.waitKey(10)

        X_input = np.array(lane_pts)
        limits = []
        X = []
        Y = []
        min_ys = []
        for i in X_input:
            min_ys.append(-i[-1][1])
        
        min_y = np.sort(min_ys)[0]
        ind_min = np.argsort(min_ys)
        self.min_x = X_input[ind_min[0]][-1][1]
        idx_y = 0
        for l in ind_min:
            limits.append(len(X_input[l]))
            if X_input[l][-1][1]<X_input[l][0][1]:
                idx_y+=1
            for i in X_input[l]:
                #if i[0]>30:
                #    continue
                X.append(i[0])
                Y.append(-1*i[1])

        #print(min_y, min_ys)
        Y = np.array(Y) - min_y
        X = np.array(X) - self.min_x
        #print(X, Y, limits)
        #print(X.shape, X)
        #print(Y.shape, Y)
        #print(limits, min_ys)
        s1 = time.time()
        ind = 0
        bounds = (0, np.inf)
        if idx_y>=2:
            bounds = ([-np.inf,-np.inf,-np.inf,0], [0,0,0,np.inf])
        #for i in range(len(limits)):
        #    y = Y[ind:int(limits[i])]
        #    for j in range(1+int(limits[i]), len(y)+int(limits[i])):
        #        y[j-int(limits[i])] = 0.95*y[j-1-int(limits[i])] + 0.05*y[j-int(limits[i])]
        #    ind = int(limits[i])
        if self.loop%4 == 0 or self.num_lanes != len(X_input):
            self.L.update_vals(X, Y, limits, min_y, bounds)
            s2 = time.time()
            print("time for curve fit= ", s2-s1)
        self.num_lanes = len(X_input)
        params = self.L.get_eqns()
        print("bounds = ", params)
        xdata_plot = np.linspace(0, 100, 220)
        #plt.plot(xdata1, ydata1)
        #plt.plot(xdata2, ydata2)
        #plt.plot(xdata3, ydata3)
        #s1 = time.time()
        #d = self.L.get_p_dist(0,0)
        #s2 = time.time()
        #print("time calc2= ", s2-s1)
        x_pose = 0
        y_pose = -self.L.centerline_params[3]
        cyy = self.L.centerline_params[3]
        self.L.centerline_params[3] = 0
        y_centerline = self.func_3d(xdata_plot-self.min_x, *self.L.centerline_params)
        #psi = -np.arctan2(y_centerline[1]-y_centerline[0], xdata_plot[1]-xdata_plot[0])
        #print(np.arctan(3*self.L.centerline_params[0]*np.power(np.array(xdata_plot),2) + 2*self.L.centerline_params[1]*np.array(xdata_plot) + self.L.centerline_params[2])*180/np.pi)
        #print("CL psi", np.arctan2(np.diff(y_centerline), np.diff(xdata_plot))*180/np.pi)

        """for pts in world_pt:
            #print(pts)
            self.ax1.scatter(pts[0], -1*pts[1]-cyy, alpha = 0.25)#'orange')"""
        for i in range(len(limits)):
            params[i][3] = params[i][3] - cyy
            ydata_plot = self.func_3d(xdata_plot-self.min_x, *params[i])
            #III = quad(self.integrand_3d, 0, 50, args=(params[i][0], params[i][1], params[i][2]))
            self.ax1.plot(xdata_plot, ydata_plot, 'b--')
        self.ax1.plot(xdata_plot, y_centerline, 'b--')
        self.ax1.scatter(np.array(all_lane_pts)[:, 0], -1*np.array(all_lane_pts)[:, 1]-cyy)
        #self.ax1.scatter(X, Y+cyy*2)

        xcoord = np.linspace(0, 100, 200)
        print("centerline params = ", self.L.centerline_params)

        s1 = time.time()
        arc_length = [quad(self.integrand_3d, 0, i, args=(self.L.centerline_params[0], self.L.centerline_params[1], self.L.centerline_params[2]))[0] for i in xcoord]
        s2 = time.time()
        #print("time = ", s1-s2)
        ycoord = self.func_3d(xcoord-self.min_x, *self.L.centerline_params)
        psicoord = np.arctan2(np.diff(ycoord), np.diff(xcoord-self.min_x))
        psi = -1*np.mean(psicoord[:3])
        #print("CL psi", psicoord*180/np.pi)

        prev = 0
        max_y = 0
        if len(self.steps)>0:
            for i in range(self.num_goal):
                path_x_coor = []
                path_y_coor = []
                path_psi = []
                slope = []
                cent_x = []
                cent_y = []
                for xc in range(2, len(self.pre_x[prev:prev+self.steps[i]])):
                    max_idx = np.where((arc_length >= self.pre_x[prev+xc]))[0]
                    if len(max_idx)<1:
                        continue
                    max_idx = max_idx[0]
                    total_arc_length_diff = arc_length[max_idx] - arc_length[max_idx-1]
                    total_x_diff = xcoord[max_idx] - xcoord[max_idx-1]
                    current_arc_length_diff = self.pre_x[prev+xc] - arc_length[max_idx-1]
                    centerline_x = xcoord[max_idx-1] + current_arc_length_diff*total_x_diff/total_arc_length_diff
                    centerline_y = self.func_3d(centerline_x-self.min_x, self.L.centerline_params[0], self.L.centerline_params[1], self.L.centerline_params[2], self.L.centerline_params[3])
                    slope_tangent = 3*self.L.centerline_params[0]*np.power(np.array(centerline_x),2) + 2*self.L.centerline_params[1]*np.array(centerline_x) + self.L.centerline_params[2]
                    theta_perpendicular = np.arctan(slope_tangent) + np.pi/2
                    x_g = centerline_x + self.pre_y[prev+xc]*np.cos(theta_perpendicular)
                    y_g = centerline_y + self.pre_y[prev+xc]*np.sin(theta_perpendicular)
                    path_x_coor.append(x_g)
                    path_y_coor.append(y_g)
                    cent_x.append(x_g)
                    cent_y.append(y_g)
                    path_psi.append(self.pre_psi[prev+xc] + np.arctan(slope_tangent))
                    slope.append(np.arctan(slope_tangent))
                    max_y = max(max_y, max(path_y_coor))
                if i == self.index:
                    self.ax1.plot(path_x_coor, path_y_coor, alpha = 1, linewidth=4.0, color='red')#'orange')
                    self.ax1.scatter(path_x_coor, path_y_coor, color='red')#'orange')
                    self.ax1.plot(cent_x, cent_y, 'b--')
                    #psi_lane = []
                    # 0::2
                    #for y in range(0, len(psicoord), 2):
                    #    psi_lane.append(psicoord[y])
                    #print("psi_lane", np.array(psi_lane)*180/np.pi)
                    #psi_lane = np.array(psi_lane) + np.array(self.pre_psi[prev:prev+self.steps[i]])
                    #print("psi_lane traj = ", psi_lane*180/np.pi)
                    #print("slope = ", np.array(slope)*180/np.pi)
                    #print("psi = ", np.array(path_psi)*180/np.pi)
                    omega = np.diff(np.array(path_psi))/self.dt
                    self.omega = 1*np.mean(omega[:3])
                else:
                    self.ax1.plot(path_x_coor, path_y_coor, alpha = 0.75, linewidth=2.0)#'orange')
                prev += self.steps[i]

        prev = 0
        if len(self.steps)>0:
            for i in range(self.num_goal):
                if i == self.index:
                    self.ax2.plot(self.pre_x[prev+2:prev+self.steps[i]], self.pre_y[prev+2:prev+self.steps[i]], alpha = 1, linewidth=4.0, color='red')#'orange')
                    self.ax2.scatter(self.pre_x[prev+self.steps[i]-1], self.pre_y[prev+self.steps[i]-1], color='red')#'orange')
                else:
                    self.ax2.plot(self.pre_x[prev+2:prev+self.steps[i]], self.pre_y[prev+2:prev+self.steps[i]], alpha = 0.75, linewidth=2.0)#, color='green')#'orange')
                prev += self.steps[i]
        self.ax1.set_xlim(-2, 115)
        self.ax2.set_xlim(-50, 95)
        self.ax1.set_ylim(-20, 20)
        self.ax2.set_ylim(-10, 10)
        #plt.draw()
        plt.pause(0.0001)
        #plt.show()
        self.L.centerline_params[3] = cyy
        return x_pose, y_pose, psi


    def projection_quadratic(self):
        """idx_y = 0
        image_rgb = self.csm.sensor_data[1]
        array = np.frombuffer(image_rgb.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image_rgb.height, image_rgb.width, 4))
        array = array[:, :, :3]
        lane_pt, world_pt = self.csm.projection_new(num=30)
        img = array.copy()
        for points in lane_pt:
            for p in points:
                if p[0]>=0 and p[1]>=0:
                    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)
        cv2.imshow("points", img)
        cv2.waitKey(1)

        X_input = np.array(world_pt)
        limits = []
        X = np.array([])
        Y = np.array([])
        min_ys = []
        for i in X_input:
            min_ys.append(-i[1][0])
        
        min_y = np.sort(min_ys)[0]
        ind_min = np.argsort(min_ys)
        for l in ind_min:
            #print(l,min_ys, min_y)
            #print(X_input[l])
            limits.append(len(X_input[l][0]))
            X = np.concatenate((X, X_input[l][0]))
            Y = np.concatenate((Y, -1*X_input[l][1]))
            if -1*X_input[l][1][0]>-1*X_input[l][1][-1]:
                idx_y+=1
        s2 = time.time()
        #print(s2-s1)"""

        image_rgb = self.csm.sensor_data[1]
        point_cloud = self.csm.sensor_data[2]
        array = np.frombuffer(image_rgb.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image_rgb.height, image_rgb.width, 4))
        array = array[:, :, :3]
        #pcd = o3d.geometry.PointCloud()

        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        num_channel_pts = []
        num_channel_pts.append(0)
        s1 = time.time()
        for i in range(point_cloud.channels):
            num_channel_pts.append(point_cloud.get_point_count(i)+num_channel_pts[-1])
        points = data[:, :-1]
        #points[:, 1] = -points[:, 1]
        channelwise_pts = []
        for i in range(64):
            channel_pt_ap = points[num_channel_pts[i]:num_channel_pts[i+1],:]
            channelwise_pts.append(channel_pt_ap)
        #channelwise_pts = [i[np.where((np.arctan2(i[:,1], i[:,0])<np.pi/4) & (np.arctan2(i[:,1], i[:,0])>-np.pi/4))[0]] for i in channelwise_pts]
        channelwise_pts = [ (self.c2lt@np.vstack((i.T, np.ones((1, i.T.shape[1])))))[:3,:].T for i in channelwise_pts]
        channelwise_pts = [ i[np.where((i[:,0]>0.0))[0]] for i in channelwise_pts]
        channelwise_pixels = [ self.c_int@i[:, [1,2,0]].T for i in channelwise_pts]
        channelwise_pixels = [ (i/i[2:,].reshape(1, i.shape[1])).T[:, :2] for i in channelwise_pixels]
        channelwise_indices = [ np.where((i[:,0]<800) & (i[:,0]>0) & (i[:,1]<600) & (i[:,1]>0))[0] for i in channelwise_pixels]
        channelwise_projection_data = [[channelwise_pts[i][channelwise_indices[i]], channelwise_pixels[i][channelwise_indices[i]]] for i in range(len(channelwise_indices))]
        idx = 0
        min_indices = []
        all_lane_pts = []
        all_lane_pix_pts = []
        s2= time.time()
        #print(s2-s1)
        lid_img2 = array.copy()
        s1 = time.time()
        all_lane_pix = self.lane_pred(self.nnet, array)
        s2 = time.time()
        #print(s2-s1)

        s1 = time.time()
        for pix_pts in channelwise_projection_data:
            pix_pts[1][:, 1] = 600 - pix_pts[1][:, 1]
        lane_pts = []
        lane_pix_pts = []
        if len(all_lane_pix)>0:
            for lane in all_lane_pix:
                lp_pt = []
                lp_pix_pt = []
                for pts in channelwise_projection_data:
                    if pts[1].shape[0] == 0 or pts[1].shape[1] == 0:
                        continue
                    if np.max(pts[1][:,1]) < np.max(np.array(lane)[:, 1]) and np.min(pts[1][:,1]) > np.min(np.array(lane)[:, 1]):
                        dists = spatial.distance.cdist(pts[1], np.array(lane))
                        min_id = np.argwhere(dists == np.min(dists))[0][0]
                        lp_pt.append(pts[0][min_id])
                        lp_pix_pt.append(pts[1][min_id])
                        all_lane_pts.append(pts[0][min_id])
                        all_lane_pix_pts.append(pts[1][min_id])
                lane_pts.append(lp_pt)
                lane_pix_pts.append(lp_pix_pt)   

        for p in all_lane_pix_pts:
            cv2.circle(lid_img2, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)
        #cv2.imshow("lanes projection", lid_img2)
        cv2.waitKey(10)

        X_input = np.array(lane_pts)
        limits = []
        X = []
        Y = []
        min_ys = []
        for i in X_input:
            min_ys.append(-i[-1][1])
        
        min_y = np.sort(min_ys)[0]
        ind_min = np.argsort(min_ys)
        #self.min_x = X_input[ind_min[0]][0][0]
        idx_y = 0
        for l in ind_min:
            limits.append(len(X_input[l]))
            if X_input[l][-1][1]<X_input[l][0][1]:
                idx_y+=1
            for i in X_input[l]:
                #if i[0]>30:
                #    continue
                X.append(i[0])
                Y.append(-1*i[1])

        #print(min_y, min_ys)
        Y = np.array(Y) - min_y
        X = np.array(X) - self.min_x
        #print(X, Y, limits)
        #print(X.shape, X)
        #print(Y.shape, Y)
        #print(limits, min_ys)
        s1 = time.time()
        ind = 0
        bounds = (0, np.inf)
        if idx_y>=2:
            bounds = ([-np.inf,-np.inf,0], [0,0,np.inf])
        #for i in range(len(limits)):
        #    y = Y[ind:int(limits[i])]
        #    for j in range(1+int(limits[i]), len(y)+int(limits[i])):
        #        y[j-int(limits[i])] = 0.95*y[j-1-int(limits[i])] + 0.05*y[j-int(limits[i])]
        #    ind = int(limits[i])
        if self.loop%1 == 0 or self.num_lanes != len(X_input):
            self.L.update_vals(X, Y, limits, min_y, bounds)
            s2 = time.time()
            print("time for curve fit= ", s2-s1)
        self.num_lanes = len(X_input)
        params = self.L.get_eqns()
        print("bounds = ", params)
        xdata_plot = np.linspace(0, 100, 220)
        #plt.plot(xdata1, ydata1)
        #plt.plot(xdata2, ydata2)
        #plt.plot(xdata3, ydata3)
        #s1 = time.time()
        #d = self.L.get_p_dist(0,0)
        #s2 = time.time()
        #print("time calc2= ", s2-s1)
        x_pose = 0
        y_pose = -self.L.centerline_params[2]
        cyy = self.L.centerline_params[2]
        self.L.centerline_params[2] = 0
        y_centerline = self.func_2d(xdata_plot-self.min_x, *self.L.centerline_params)
        #psi = -np.arctan2(y_centerline[1]-y_centerline[0], xdata_plot[1]-xdata_plot[0])
        #print(np.arctan(3*self.L.centerline_params[0]*np.power(np.array(xdata_plot),2) + 2*self.L.centerline_params[1]*np.array(xdata_plot) + self.L.centerline_params[2])*180/np.pi)
        #print("CL psi", np.arctan2(np.diff(y_centerline), np.diff(xdata_plot))*180/np.pi)

        """for pts in world_pt:
            #print(pts)
            self.ax1.scatter(pts[0], -1*pts[1]-cyy, alpha = 0.5)#'orange')"""
        for i in range(len(limits)):
            params[i][2] = params[i][2] - cyy
            ydata_plot = self.func_2d(xdata_plot-self.min_x, *params[i])
            #III = quad(self.integrand_3d, 0, 50, args=(params[i][0], params[i][1], params[i][2]))
            self.ax1.plot(xdata_plot, ydata_plot, 'b--')
        self.ax1.plot(xdata_plot, y_centerline, 'b--')
        self.ax1.scatter(np.array(all_lane_pts)[:, 0], -1*np.array(all_lane_pts)[:, 1]-cyy)
        #self.ax1.scatter(X, Y+cyy*2)

        xcoord = np.linspace(0, 100, 200)
        print("centerline params = ", self.L.centerline_params)

        s1 = time.time()
        arc_length = [quad(self.integrand_2d, 0, i, args=(self.L.centerline_params[0], self.L.centerline_params[1]))[0] for i in xcoord]
        s2 = time.time()
        #print("time = ", s1-s2)
        ycoord = self.func_2d(xcoord-self.min_x, *self.L.centerline_params)
        psicoord = np.arctan2(np.diff(ycoord), np.diff(xcoord-self.min_x))
        psi = -1*np.mean(psicoord[:3])
        #print("CL psi", psicoord*180/np.pi)

        prev = 0
        max_y = 0
        if len(self.steps)>0:
            for i in range(self.num_goal):
                path_x_coor = []
                path_y_coor = []
                path_psi = []
                slope = []
                cent_x = []
                cent_y = []
                for xc in range(2, len(self.pre_x[prev:prev+self.steps[i]])):
                    max_idx = np.where((arc_length >= self.pre_x[prev+xc]))[0]
                    if len(max_idx)<1:
                        continue
                    max_idx = max_idx[0]
                    total_arc_length_diff = arc_length[max_idx] - arc_length[max_idx-1]
                    total_x_diff = xcoord[max_idx] - xcoord[max_idx-1]
                    current_arc_length_diff = self.pre_x[prev+xc] - arc_length[max_idx-1]
                    centerline_x = xcoord[max_idx-1] + current_arc_length_diff*total_x_diff/total_arc_length_diff
                    centerline_y = self.func_2d(centerline_x-self.min_x, self.L.centerline_params[0], self.L.centerline_params[1], self.L.centerline_params[2])
                    slope_tangent = 2*self.L.centerline_params[0]*np.array(centerline_x) + self.L.centerline_params[1]
                    theta_perpendicular = np.arctan(slope_tangent) + np.pi/2
                    x_g = centerline_x + self.pre_y[prev+xc]*np.cos(theta_perpendicular)
                    y_g = centerline_y + self.pre_y[prev+xc]*np.sin(theta_perpendicular)
                    path_x_coor.append(x_g)
                    path_y_coor.append(y_g)
                    cent_x.append(x_g)
                    cent_y.append(y_g)
                    path_psi.append(self.pre_psi[prev+xc] + np.arctan(slope_tangent))
                    slope.append(np.arctan(slope_tangent))
                    max_y = max(max_y, max(path_y_coor))
                if i == self.index:
                    self.ax1.plot(path_x_coor, path_y_coor, alpha = 1, linewidth=4.0, color='red')#'orange')
                    self.ax1.scatter(path_x_coor, path_y_coor, color='red')#'orange')
                    self.ax1.plot(cent_x, cent_y, 'b--')
                    #psi_lane = []
                    # 0::2
                    #for y in range(0, len(psicoord), 2):
                    #    psi_lane.append(psicoord[y])
                    #print("psi_lane", np.array(psi_lane)*180/np.pi)
                    #psi_lane = np.array(psi_lane) + np.array(self.pre_psi[prev:prev+self.steps[i]])
                    #print("psi_lane traj = ", psi_lane*180/np.pi)
                    #print("slope = ", np.array(slope)*180/np.pi)
                    print("psi = ", np.array(path_psi)*180/np.pi)
                    omega = np.diff(np.array(path_psi))/self.dt
                    self.omega = 1*np.mean(omega[:7])
                else:
                    self.ax1.plot(path_x_coor, path_y_coor, alpha = 0.75, linewidth=2.0)#'orange')
                prev += self.steps[i]

        prev = 0
        if len(self.steps)>0:
            for i in range(self.num_goal):
                if i == self.index:
                    self.ax2.plot(self.pre_x[prev+2:prev+self.steps[i]], self.pre_y[prev+2:prev+self.steps[i]], alpha = 1, linewidth=4.0, color='red')#'orange')
                    self.ax2.scatter(self.pre_x[prev+self.steps[i]-1], self.pre_y[prev+self.steps[i]-1], color='red')#'orange')
                else:
                    self.ax2.plot(self.pre_x[prev+2:prev+self.steps[i]], self.pre_y[prev+2:prev+self.steps[i]], alpha = 0.75, linewidth=2.0)#, color='green')#'orange')
                prev += self.steps[i]
        self.ax1.set_xlim(-2, 105)
        self.ax2.set_xlim(-50, 95)
        self.ax1.set_ylim(-20, 20)
        self.ax2.set_ylim(-10, 10)
        #plt.draw()
        plt.pause(0.0001)
        #plt.show()
        self.L.centerline_params[2] = cyy
        return x_pose, y_pose, psi
        


        
    def timer_callback(self):

        if (self.obs[0][0] < 600 and self.setting == 2) or (self.obs[0][1] >= -8.0-0.7 and self.setting == 1) or (self.obs[0][0] < 1000 and self.setting == 0) or (self.obs[0][0] < 400 and self.NGSIM == True):#self.obs[0][1] >= -8.0-0.7:#self.obs[0][0] < 1000:#self.obs[0][1] >= -8.0-0.7:#self.obs[0][0] < 1000 or self.obs[0][1] >= -10:
            
            if self.Gotit or self.flag:
                #t1 = time()
                dt = self.dt
                
                #self.ours_x.append(self.obs[0][0])
                #self.ours_y.append(self.obs[0][1])

                self.loop += 1
                #self.sim_time = np.append(self.sim_time, self.loop * dt)

                self.csm.pre_x = self.pre_x
                self.csm.pre_y = self.pre_y
                self.csm.pre_psi = self.pre_psi
                self.csm.sel_index = self.index
                self.csm.steps = self.steps
                
                        
                self.lower_lim = -30 + self.obs[0][0]
                self.upper_lim = self.upper + self.obs[0][0]


                self.csm.visualize()
                acc = (self.v-self.prev_vel)/dt
                prev = 0
                selected_psi = []
                selected_x = []
                selected_y = []
                omega = self.w
                if len(self.steps)>0:
                    p = []
                    prev = 0
                    for i in range(self.num_goal):
                        if i == self.index:
                            selected_x = self.pre_x[prev+0:prev+self.steps[i]]
                            selected_y = self.pre_y[prev+0:prev+self.steps[i]]
                        prev += self.steps[i]
                    selected_psi = np.array(self.csm.pre_sel_psi)*2*np.pi/360
                    #print(self.csm.pre_sel_psi)
                    #print(p, end=" ")
                    #print(selected_psi)
                    #prev += self.steps[i]
                    psi_dot = np.diff(selected_psi)/dt
                    #print(psi_dot)
                    omega = -1*np.mean(psi_dot[:3])
                    #self.csm.save_data(self.loop-1, selected_x[:23], selected_y[:23], self.v)
                    #print(omega, op)
                plt.clf()                        
                #self.ax = self.fig.add_subplot(111, aspect='equal')
                self.ax1 = self.fig.add_subplot(211, aspect='equal')
                self.ax2 = self.fig.add_subplot(212, aspect='equal')
                prev = 0
                if len(self.steps)>0:
                    
                    """for i in range(self.num_goal):
                        if i == self.index:
                            plt.plot(self.pre_x[prev+2:prev+self.steps[i]], self.pre_y[prev+2:prev+self.steps[i]], alpha = 1, linewidth=4.0, color='red')#'orange')
                            plt.scatter(self.pre_x[prev+self.steps[i]-1], self.pre_y[prev+self.steps[i]-1], color='red')#'orange')
                        else:
                            plt.plot(self.pre_x[prev+2:prev+self.steps[i]], self.pre_y[prev+2:prev+self.steps[i]], alpha = 0.75, linewidth=2.0)#, color='green')#'orange')
                        prev += self.steps[i]"""
                    
                    
                    
                    diag = np.sqrt(self.a_rect ** 2 + self.b_rect ** 2)
                    ells = [Ellipse(xy=[self.other_vehicles[i][0], self.other_vehicles[i][1]], width=self.a_ell, height=self.b_ell, angle=0.0) for i in range(len(self.other_vehicles))]
                    #print(len(ells), len(rect), len())
                    mm = 0

                    # Ground Truth obs coordinates
                    """for e in ells:
                        self.ax2.add_artist(e)
                        #self.ax1.add_artist(r)
                        #self.ax2.add_artist(e)
                        #self.ax1.add_artist(Rectangle(xy=[self.other_vehicles_cl[mm][0] -  diag/2 * np.cos(0 + np.arctan(self.b_rect/self.a_rect)), self.other_vehicles_cl[mm][1] - diag/2 * np.sin(0 + np.arctan(self.b_rect/self.a_rect))], width=4.0, height=1.4, angle=self.other_vehicles_cl[mm][2]*180.0/np.pi))
                        self.ax1.plot([self.other_vehicles_cl[mm][0][0,0],self.other_vehicles_cl[mm][1][0,0]],[self.other_vehicles_cl[mm][0][1,0]+self.obs[0][1] ,self.other_vehicles_cl[mm][1][1,0]+self.obs[0][1]],'r')
                        self.ax1.plot([self.other_vehicles_cl[mm][1][0,0],self.other_vehicles_cl[mm][2][0,0]],[self.other_vehicles_cl[mm][1][1,0]+self.obs[0][1] ,self.other_vehicles_cl[mm][2][1,0]+self.obs[0][1]],'r')
                        self.ax1.plot([self.other_vehicles_cl[mm][2][0,0],self.other_vehicles_cl[mm][3][0,0]],[self.other_vehicles_cl[mm][2][1,0]+self.obs[0][1] ,self.other_vehicles_cl[mm][3][1,0]+self.obs[0][1]],'r')
                        self.ax1.plot([self.other_vehicles_cl[mm][0][0,0],self.other_vehicles_cl[mm][3][0,0]],[self.other_vehicles_cl[mm][0][1,0]+self.obs[0][1] ,self.other_vehicles_cl[mm][3][1,0]+self.obs[0][1]],'r')
                        e.set_clip_box(self.ax2.bbox)
                        e.set_alpha(1)
                        e.set_facecolor([0.0, 0.5, 1])
                        #print(mm)
                        # if mm < 6:
                        #     e.set_facecolor([0.5, 0.5, 0.5])
                        #if (self.other_vehicles[mm][0] < self.upper_lim - 2.5 and self.other_vehicles[mm][0] > self.lower_lim + 2.5) and (self.other_vehicles[mm][1] > -12 and self.other_vehicles[mm][1] < 12):
                        #self.ax1.text(self.other_vehicles[mm][0]-2, self.other_vehicles[mm][1]-0.3, '%s'%(round((self.other_vehicles[mm][2]**2 + self.other_vehicles[mm][3]**2)**0.5,2)), fontsize=10)
                        mm+=1"""

                    # Lidar vehicle coordinates
                    for e in ells:
                        self.ax2.add_artist(e)
                        #self.ax1.add_artist(r)
                        #self.ax2.add_artist(e)
                        #self.ax1.add_artist(Rectangle(xy=[self.other_vehicles_cl[mm][0] -  diag/2 * np.cos(0 + np.arctan(self.b_rect/self.a_rect)), self.other_vehicles_cl[mm][1] - diag/2 * np.sin(0 + np.arctan(self.b_rect/self.a_rect))], width=4.0, height=1.4, angle=self.other_vehicles_cl[mm][2]*180.0/np.pi))
                        self.ax1.plot( [self.other_vehicles_cl[mm][0], self.other_vehicles_cl[mm][0]], [self.other_vehicles_cl[mm][2], self.other_vehicles_cl[mm][3]],'r')
                        self.ax1.plot( [self.other_vehicles_cl[mm][0], self.other_vehicles_cl[mm][1]], [self.other_vehicles_cl[mm][2], self.other_vehicles_cl[mm][2]],'r')
                        self.ax1.plot( [self.other_vehicles_cl[mm][1], self.other_vehicles_cl[mm][1]], [self.other_vehicles_cl[mm][2], self.other_vehicles_cl[mm][3]],'r')
                        self.ax1.plot( [self.other_vehicles_cl[mm][0], self.other_vehicles_cl[mm][1]], [self.other_vehicles_cl[mm][3], self.other_vehicles_cl[mm][3]],'r')
                        e.set_clip_box(self.ax2.bbox)
                        e.set_alpha(1)
                        e.set_facecolor([0.0, 0.5, 1])
                        #print(mm)
                        # if mm < 6:
                        #     e.set_facecolor([0.5, 0.5, 0.5])
                        #if (self.other_vehicles[mm][0] < self.upper_lim - 2.5 and self.other_vehicles[mm][0] > self.lower_lim + 2.5) and (self.other_vehicles[mm][1] > -12 and self.other_vehicles[mm][1] < 12):
                        #self.ax1.text(self.other_vehicles[mm][0]-2, self.other_vehicles[mm][1]-0.3, '%s'%(round((self.other_vehicles[mm][2]**2 + self.other_vehicles[mm][3]**2)**0.5,2)), fontsize=10)
                        mm+=1
                    
                    
                    rob = [Ellipse(xy=[self.obs[0][0], self.obs[0][1]], width=self.a_ell, height=self.b_ell, angle=0*self.psi*180.0/np.pi)]
                    rect = [Rectangle(xy=[self.obs[0][0] - diag/2 * np.cos(self.psi + np.arctan(self.b_rect/self.a_rect)), self.obs[0][1] - diag/2 * np.sin(self.psi + np.arctan(self.b_rect/self.a_rect))], width=5.0, height=2.5, angle=1*self.psi*180.0/np.pi)]
                    for e, r in zip(rob, rect):
                        self.ax1.add_artist(e)
                        #self.ax2.add_artist(r)
                        self.ax2.add_patch(Rectangle(xy=[self.obs[0][0] - diag/2 * np.cos(self.psi + np.arctan(self.b_rect/self.a_rect)), self.obs[0][1] - diag/2 * np.sin(self.psi + np.arctan(self.b_rect/self.a_rect))+0.7], width=4.0, height=1.4, angle=1*self.psi*180.0/np.pi))
                        #self.ax1.add_artist(r)
                        #self.ax2.add_artist(e)
                        e.set_clip_box(self.ax1.bbox)
                        #e.set_clip_box(self.ax2.bbox)
                        e.zorder = 10
                        r.zorder = 10
                        e.set_alpha(1)
                        r.set_alpha(0.5)
                        r.set_facecolor([1, 1, 1])
                        e.set_facecolor([1, 0.5, 0.5])
                        # e.set_facecolor([1.0, 0.647, 0.0]) # orange
                        # e.set_facecolor([0, 100/255, 0]) #green
                        self.ax1.text(self.obs[0][0]-2, self.obs[0][1]-0.3, '%s'%(round((self.obs[0][2]**2 + self.obs[0][3]**2)**0.5,2)), fontsize=10, zorder = 20)
                    
                    for i in range(1, len(self.x_prev)):
                        #print(len(self.x_prev), self.x_prev[i], -(len(self.x_prev)-i)*4.5, i)
                        self.ax2.add_patch(Rectangle(xy=[-(len(self.x_prev)-i)*4.5 - diag/2 * np.cos(self.x_prev[i][2] + np.arctan(self.b_rect/self.a_rect)), self.x_prev[i][1] - diag/2 * np.sin(self.x_prev[i][2] + np.arctan(self.b_rect/self.a_rect))+0.7], width=4.0, height=1.4, angle=1*self.x_prev[i][2]*180.0/np.pi))
                        r.zorder = 10
                        r.set_alpha(0.5)
                        r.set_facecolor([1.0, 0.647, 0.0])
                    
                    
                    #plt.plot(self.ours_x, self.ours_y, color=[1, 0.5, 0.5])

                    #plt.xlabel('Y in m')
                    #plt.ylabel('X in m')
                    self.lower_lim = -100
                    self.upper_lim = 100
                    
                    
                    if self.flag == 0:
                        #plt.text(self.lower_lim+10, 14, 'Collision with obstacle= %s'%((self.intersection)), fontsize=10) 
                        self.ax2.text(self.lower_lim+30, 10, 'Carla Orientation= %s degrees'%(round(self.psi_act*180/np.pi, 3)), fontsize=20)
                        self.ax2.text(self.lower_lim+100, 10, 'Centerline Orientation= %s degrees'%(round(self.psi*180/np.pi, 3)), fontsize=20)
                        self.ax2.text(self.lower_lim+170, 10, 'B= %s'%(round(self.L.centerline_params[1],3)), fontsize=20)
                        
                    #self.ax1.text((self.lower_lim+self.upper_lim)/2 - 15, 20, 'Highway environment', fontsize=14)
                    self.ax2.plot([self.lower_lim, self.upper_lim], [-3.5, -3.5], color='black',linestyle='-', alpha=0.2)
                    self.ax2.plot([self.lower_lim, self.upper_lim], [0, 0], color='black',linestyle='--', alpha=0.2)
                    self.ax2.plot([self.lower_lim, self.upper_lim], [3.5, 3.5], color='black',linestyle='-', alpha=0.2)
                    #plt.plot([self.lower_lim, self.upper_lim], [-7, -7], color='black',linestyle='-', alpha=0.2)
                    #plt.plot([self.lower_lim, self.upper_lim], [7, 7], color='black',linestyle='-', alpha=0.2)
                    #plt.xlim(self.lower_lim, self.upper_lim)
                    #plt.ylim(-12, 12)
                    # plt.title('highway env')

                    #plt.tight_layout()
                    # plt.draw()
                    #plt.pause(0.000000000000000001)
                    #plt.pause(100)
                print("Velocity = ", self.v, " W ", self.w, " Omega = ", self.omega, " Acc = ", acc, "PrevAcc = ", self.prev_acc)
                #8 obstacle mode = self.csm.apply_control(self.v, omega, acc, self.prev_acc)
                self.csm.apply_control(self.v, self.omega, acc, self.prev_acc)
                t1 = time.time()
                self.csm.simulate()
                #print("sim_done = ", self.loop)
                vel, obs, bb = self.csm.global_to_centerline()
                t2 = time.time()
                #print("tc1 = = = = ", t2-t1)
                #_1, _2, psip = self.projection_linear()
                #xp,yp,psip = self.projection(self.pre_x, self.pre_y)
                t1 = time.time()
                #xp,yp,psip = self.projection_cubic()#self.projection_new()
                xp,yp,psip = self.projection_quadratic()
                #xp,yp,psip = self.projection_exp()
                t2 = time.time()
                #print("tc = = = = ", t2-t1)
                self.x_prev.append([xp, yp, psip])
                #ego = [xp, yp, psip, vel[2], vel[3]]
                #psip = 0.95*self.mv_psi + 0.05*psip
                self.mv_psi = psip
                ego = [xp, yp, psip, vel[2], vel[3]]
                #print("ego = ", ego)
                #print(obs)
                #ego[2] = ego[2] * 2*np.pi/360
                self.psi = ego[2]
                self.psi_act = vel[1]
                #print(self.psi, self.psi*180/np.pi)
                obs = np.array(obs)
                #obs = obs[obs[:, 4].argsort()]
                #print(obs)
                
                xcoord = np.linspace(-50, 150, 800)
                ycoord = self.func_2d(xcoord-self.min_x, *self.L.centerline_params)
                psicoord = np.arctan2(np.diff(ycoord), np.diff(xcoord))
                self.other_vehicles_cl = []
                for i in range(len(obs)):
                    print("obs1 = ", obs[i])
                    self.other_vehicles_cl.append(bb[i])
                    if obs[i][0]<1000:
                        dists = ((obs[i][0] - xcoord)**2 + (obs[i][1] - ycoord)**2)**0.5
                        min_dist = np.argmin(dists)
                        y_cl = dists[min_dist] if obs[i][1]>ycoord[min_dist] else -dists[min_dist]
                        x_cl = xcoord[min_dist]
                        x_sl = quad(self.integrand_2d, 0, x_cl, args=(self.L.centerline_params[0], self.L.centerline_params[1]))[0]
                        obs[i][0] = x_sl
                        obs[i][1] = y_cl
                        
                        obs[i][2] = (obs[i][2]**2 + obs[i][3]**2)**0.5
                        obs[i][3] = 0.0
                    print("obs2 = ", obs[i])
                    """if pos >= 6:
                        continue
                    if obs[i][0]>0:
                        #print(obs[i])
                        self.obs[1+pos] = obs[i,:4]
                        pos+=1"""
                #self.obs[1:] = obs[:len(self.obs)-1,:4]
                #self.other_vehicles = self.obs[1:]
                #print("ov = ", self.other_vehicles_cl)
                self.other_vehicles = obs
                obs = obs[obs[:, 4].argsort()]
                pos = 0
                for i in range(len(obs)):
                    if pos >= 6:
                        continue
                    if obs[i][0]>0:
                        #print(obs[i])
                        self.obs[1+pos] = obs[i,:4]
                        pos+=1

                self.obs[0] = [ego[0], ego[1], ego[3], ego[4]]
                self.prev_vel = ((ego[3])**2 + (ego[4])**2)**0.5
                self.prev_acc = acc
                print("self.obs = ", self.obs)
                #self.checkCollision()
                msg = States()
                msg.x = self.obs[:,0].T.tolist()
                msg.y = self.obs[:,1].T.tolist()
                msg.vx = self.obs[:,2].T.tolist()
                msg.vy = self.obs[:,3].T.tolist()
                msg.psi = (self.psi * np.ones(5)).tolist()
                gx = []
                gy = []
                msg.num_goal = 1#(self.num_lanes-1)*4
                lval = 0.0#(self.num_lanes-2)*3.5/2
                for i in range(1):
                    gy.append(lval)
                    #gy.append(lval)
                    #gy.append(lval)
                    #gy.append(lval)
                    gx.append(60.0 + 20*(self.num_lanes-2))
                    #gx.append(50.0 + 20*(self.num_lanes-2))
                    #gx.append(60.0 + 20*(self.num_lanes-2))
                    #gx.append(70.0 + 20*(self.num_lanes-2))
                    lval = lval-3.5
                msg.goal_x = gx
                msg.goal_y = gy

                msg.psidot = ((msg.psi[0] - self.prev_psi)/dt) 
                #print(msg.x)
                
                self.prev_psi = msg.psi[0]
                self.publisher_.publish(msg)
                self.csm.vis_path = True

                self.Gotit = 0
                self.flag = 0
                #print("teleporting")
                self.csm.teleport_obstacles(self.other_vehicles)
                x = Symbol('x')
                t = Symbol('t')
                xcoord = np.linspace(0, 100, 200)
                #arc_length = [quad(self.integrand, 0, i, args=(self.L.centerline_params[0], self.L.centerline_params[1]))[0] for i in xcoord]
                #for i in range(len(self.other_vehicles)):
                #    self.other_vehicles[i][0] = solve(Eq(fx, i), t)[-1]
                    

                

def main(args=None):
    rclpy.init(args=args)
    pygame.init()
    display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = cbs.get_font()
    clock = pygame.time.Clock()
    minimal_subscriber = MinimalSubscriber(display, font, clock)
    try:
        rclpy.spin(minimal_subscriber)
    except (KeyboardInterrupt, IndexError, AttributeError) as e:
        minimal_subscriber.csm.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



