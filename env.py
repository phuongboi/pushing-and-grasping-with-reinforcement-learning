#!/usr/bin/env python

import rospy
import math
import random
import numpy as np
from collections import deque
from std_msgs.msg import Int8MultiArray, Float32, Bool
from geometry_msgs.msg import Transform
import os
from simulation import vrep
import time
import utils
import cv2

class VrepEnvironment():
    def __init__(self, is_testing):
        self.is_sim = True
        self.is_testing = is_testing
        self.obj_mesh_dir = os.path.abspath('objects/blocks/')
        self.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
        self.heightmap_resolution = 0.004#0.002 # use 4mm resolution to reduce computation
        self.action_space = [112, 112, 16]
        self.num_rotations = 16
        self.no_change_count = [2, 2] if not is_testing else [0, 0]

        # Define colors for object meshes (Tableau palette)
        self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                       [89.0, 161.0, 79.0], # green
                                       [156, 117, 95], # brown
                                       [242, 142, 43], # orange
                                       [237.0, 201.0, 72.0], # yellow
                                       [186, 176, 172], # gray
                                       [255.0, 87.0, 89.0], # red
                                       [176, 122, 161], # purple
                                       [118, 183, 178], # cyan
                                       [255, 157, 167]])/255.0 #pink

        # Read files in object mesh directory
        self.num_obj = 10
        self.mesh_list = os.listdir(self.obj_mesh_dir)

        # Randomly choose objects to add to scene
        self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
        self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]

        # Make sure to have the server side running in V-REP:
        # in a child script of a V-REP scene, add following command
        # to be executed just once, at simulation start:
        #
        # simExtRemoteApiStart(19999)
        #
        # then start simulation, and run this program.
        #
        # IMPORTANT: for each successful call to simxStart, there
        # should be a corresponding call to simxFinish at the end!

        # MODIFY remoteApiConnections.txt

        # Connect to simulator
        vrep.simxFinish(-1) # Just in case, close all opened connections
        self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation.')
            self.restart_sim()

        self.is_testing = is_testing
        self.test_preset_cases = True
        self.test_preset_file = "simulation/test_cases/test-10-obj-01.txt"

        # Setup virtual camera in simulation
        self.setup_sim_camera()
        # If testing, read object meshes and poses from test case file
        if self.is_testing and self.test_preset_cases:
            file = open(self.test_preset_file, 'r')
            file_content = file.readlines()
            self.test_obj_mesh_files = []
            self.test_obj_mesh_colors = []
            self.test_obj_positions = []
            self.test_obj_orientations = []
            for object_idx in range(self.num_obj):
                file_content_curr_object = file_content[object_idx].split()
                self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir,file_content_curr_object[0]))
                self.test_obj_mesh_colors.append([float(file_content_curr_object[1]),float(file_content_curr_object[2]),float(file_content_curr_object[3])])
                self.test_obj_positions.append([float(file_content_curr_object[4]),float(file_content_curr_object[5]),float(file_content_curr_object[6])])
                self.test_obj_orientations.append([float(file_content_curr_object[7]),float(file_content_curr_object[8]),float(file_content_curr_object[9])])
            file.close()
            self.obj_mesh_color = np.asarray(self.test_obj_mesh_colors)

        # Add objects to simulation environment
        self.add_objects()
    def setup_sim_camera(self):

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        cam_trans = np.eye(4,4)
        cam_trans[0:3,3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4,4)
        cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale


    def add_objects(self):

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        sim_obj_handles = []
        for object_idx in range(len(self.obj_mesh_ind)):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
            if self.is_testing and self.test_preset_cases:
                curr_mesh_file = self.test_obj_mesh_files[object_idx]
            curr_shape_name = 'shape_%02d' % object_idx
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            if self.is_testing and self.test_preset_cases:
                object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1], self.test_obj_positions[object_idx][2]]
                object_orientation = [self.test_obj_orientations[object_idx][0], self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]
            object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]
            ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)
            if not (self.is_testing and self.test_preset_cases):
                time.sleep(2)
        self.prev_obj_positions = []
        self.obj_positions = []


    def restart_sim(self):

        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.4: # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)


    def check_sim(self):

        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - 0.1 and gripper_position[0] < self.workspace_limits[0][1] + 0.1 and gripper_position[1] > self.workspace_limits[1][0] - 0.1 and gripper_position[1] < self.workspace_limits[1][1] + 0.1 and gripper_position[2] > self.workspace_limits[2][0] and gripper_position[2] < self.workspace_limits[2][1]
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.add_objects()

    def get_camera_data(self):

        if self.is_sim:

            # Get color image from simulation
            sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
            color_img = np.asarray(raw_image)
            color_img.shape = (resolution[1], resolution[0], 3)
            color_img = color_img.astype(np.float)/255
            color_img[color_img < 0] += 1
            color_img *= 255
            color_img = np.fliplr(color_img)
            color_img = color_img.astype(np.uint8)

            # Get depth image from simulation
            sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
            depth_img = np.asarray(depth_buffer)
            depth_img.shape = (resolution[1], resolution[0])
            depth_img = np.fliplr(depth_img)
            zNear = 0.01
            zFar = 10
            depth_img = depth_img * (zFar - zNear) + zNear

        return color_img, depth_img


    def close_gripper(self, asynch=False):

        if self.is_sim:
            gripper_motor_velocity = -0.5
            gripper_motor_force = 100
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
            gripper_fully_closed = False
            while gripper_joint_position > -0.045: # Block until gripper is fully closed
                sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
                # print(gripper_joint_position)
                if new_gripper_joint_position >= gripper_joint_position:
                    return gripper_fully_closed
                gripper_joint_position = new_gripper_joint_position
            gripper_fully_closed = True

        return gripper_fully_closed

    def open_gripper(self, asynch=False):

        if self.is_sim:
            gripper_motor_velocity = 0.5
            gripper_motor_force = 20
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
            while gripper_joint_position < 0.03: # Block until gripper is fully open
                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)

    def move_to(self, tool_position, tool_orientation):

        if self.is_sim:

            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)

            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]], dtype=object)
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.02*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_magnitude/0.02))

            for step_iter in range(num_move_steps):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client,self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)


    def grasp(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))

        if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

            # Avoid collision with floor
            position = np.asarray(position, dtype=object).copy()
            position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02)

            # Move gripper to location above grasp target
            grasp_location_margin = 0.15
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            location_above_grasp_target = (np.float(position[0]), np.float(position[1]), np.float(position[2] + grasp_location_margin))

            # Compute gripper position and linear movement increments
            tool_position = location_above_grasp_target
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]], dtype=object)

            move_magnitude = np.linalg.norm(move_direction)
            #print(move_magnitude)
            move_step = 0.05*move_direction/(move_magnitude+0.000001)
            #print(move_step)
            num_move_steps = int(np.floor(move_direction[0]/(move_step[0]+0.000001)))

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/(rotation_step+0.000001)))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

            # Ensure gripper is open
            self.open_gripper()

            # Approach grasp target
            self.move_to(position, None)

            # Close gripper to grasp target
            gripper_full_closed = self.close_gripper()

            # Move gripper to location above grasp target
            self.move_to(location_above_grasp_target, None)

            # Check if grasp is successful
            gripper_full_closed = self.close_gripper()
            grasp_success = not gripper_full_closed

            # Move the grasped object elsewhere
            if grasp_success:
                object_positions = np.asarray(self.get_obj_positions())
                object_positions = object_positions[:,2]
                grasped_object_ind = np.argmax(object_positions)
                grasped_object_handle = self.object_handles[grasped_object_ind]
                vrep.simxSetObjectPosition(self.sim_client,grasped_object_handle,-1,(-0.5, 0.5 + 0.05*float(grasped_object_ind), 0.1),vrep.simx_opmode_blocking)

        return grasp_success


    def push(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))

        if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

            # Adjust pushing point to be on tip of finger
            position[2] = position[2] + 0.026

            # Compute pushing direction
            push_orientation = [1.0,0.0]
            push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle), push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle)], dtype=object)

            # Move gripper to location above pushing point
            pushing_point_margin = 0.1
            location_above_pushing_point = (np.float(position[0]), np.float(position[1]), np.float(position[2] + pushing_point_margin))

            # Compute gripper position and linear movement increments
            tool_position = location_above_pushing_point
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]], dtype=object)
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05*move_direction/(move_magnitude +0.000001)
            num_move_steps = int(np.floor(move_direction[0]/(move_step[0] +0.000001)))

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/(rotation_step+0.000001)))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

            # Ensure gripper is closed
            self.close_gripper()

            # Approach pushing point
            self.move_to(position, None)

            # Compute target location (push to the right)
            push_length = 0.1
            target_x = min(max(position[0] + push_direction[0]*push_length, workspace_limits[0][0]), workspace_limits[0][1])
            target_y = min(max(position[1] + push_direction[1]*push_length, workspace_limits[1][0]), workspace_limits[1][1])
            push_length = np.sqrt(np.power(target_x-position[0],2)+np.power(target_y-position[1],2))

            # Move in pushing direction towards target location
            self.move_to([target_x, target_y, position[2]], None)

            # Move gripper to location above grasp target
            self.move_to([target_x, target_y, location_above_pushing_point[2]], None)

            push_success = True

        return push_success

    def get_obj_positions(self):

        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

    def step(self, action, prev_valid_depth_heightmap, prev_depth_heightmap):
        #print("action step")
        action_idx = np.unravel_index(action, self.action_space)
    #    action_idx = action
        #print("action ind", action_idx)
        pos_y, pos_x= action_idx[0], action_idx[1]
        rot_angle = np.deg2rad(action_idx[2]*(360.0*2/self.num_rotations)) # just 8 orientation
        primitive_position = [pos_x * self.heightmap_resolution + self.workspace_limits[0][0], pos_y * self.heightmap_resolution + self.workspace_limits[1][0], prev_valid_depth_heightmap[pos_y][pos_x] + self.workspace_limits[2][0]]

        if action_idx[2] <= 7: # define as push action
            finger_width = 0.02
            safe_kernel_width = int(np.round((finger_width/2)/self.heightmap_resolution))
            local_region = prev_valid_depth_heightmap[max(pos_y - safe_kernel_width, 0):min(pos_y + safe_kernel_width + 1, prev_valid_depth_heightmap.shape[0]), max(pos_x - safe_kernel_width, 0):min(pos_x + safe_kernel_width + 1, prev_valid_depth_heightmap.shape[1])]
            if local_region.size == 0:
                safe_z_position = self.workspace_limits[2][0]
            else:
                safe_z_position = np.max(local_region) + self.workspace_limits[2][0]
            primitive_position[2] = safe_z_position
        # fix robot out of space
        if primitive_position[2] > 0.1:
            primitive_position[2] = 0.001


        # Initialize variables that influence reward
        push_success = False
        grasp_success = False
        change_detected = False
        terminal = False

        # Execute primitive
        if action_idx[2] <= 7:
            push_success = self.push(primitive_position, rot_angle, self.workspace_limits)
            print('Push successful: %r' % (push_success))
        else:
            grasp_success = self.grasp(primitive_position, rot_angle, self.workspace_limits)
            print('Grasp successful: %r' % (grasp_success))

        color_img, depth_img = self.get_camera_data()
        depth_img = depth_img * self.cam_depth_scale # Apply depth scale from calibration
        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, self.cam_intrinsics, self.cam_pose, self.workspace_limits, self.heightmap_resolution/2)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > 0.02] = 1
        empty_threshold = 300
        if self.is_sim and self.is_testing:
            empty_threshold = 10
        if np.sum(stuff_count) < empty_threshold or (self.is_sim and self.no_change_count[0] + self.no_change_count[1] > 10):
            self.no_change_count = [0, 0]
            if self.is_sim:
                print('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
                # self.restart_sim()
                # self.add_objects()
                terminal = True
                # if is_testing: # If at end of test run, re-load original weights (before test run)
                #     trainer.model._load_state_dict(torch.load(snapshot_file))
        # Detect changes
        depth_diff = abs(depth_heightmap - prev_depth_heightmap)
        depth_diff[np.isnan(depth_diff)] = 0
        depth_diff[depth_diff > 0.3] = 0
        depth_diff[depth_diff < 0.01] = 0
        depth_diff[depth_diff > 0] = 1
        change_threshold = 800
        change_value = np.sum(depth_diff)
        change_detected = change_value > change_threshold or grasp_success
        print('Change detected: %r (value: %d)' % (change_detected, change_value))

        if change_detected:
            if action_idx[2] <= 7:
                self.no_change_count[0] = 0
            else:
                self.no_change_count[1] = 0
        else:
            if action_idx[2] <= 7:
                self.no_change_count[0] += 1
            else:
                self.no_change_count[1] += 1
        reward = 0.0
        if grasp_success:
            reward = 15.0
        else:
            if change_detected:
                reward = 5.0
        state = [color_heightmap, valid_depth_heightmap, depth_heightmap]
        #print("color heightmap shape",color_heightmap.shape)
        return state, reward, terminal

    def twohead_grasp_step(self, action, prev_valid_depth_heightmap, prev_depth_heightmap):
        action_location_idx, action_orientation_idx = action

        pos_y, pos_x= np.unravel_index(action_location_idx, self.action_space[:2])
        rot_angle = np.deg2rad(action_orientation_idx*(360.0/self.num_rotations))
        primitive_position = [pos_x * self.heightmap_resolution + self.workspace_limits[0][0], pos_y * self.heightmap_resolution + self.workspace_limits[1][0], prev_valid_depth_heightmap[pos_y][pos_x] + self.workspace_limits[2][0]]

        # fix robot out of space
        if primitive_position[2] > 0.1:
            primitive_position[2] = 0.001


        # Initialize variables that influence reward
        push_success = False
        grasp_success = False
        change_detected = False
        terminal = False

        # Execute primitive

        grasp_success = self.grasp(primitive_position, rot_angle, self.workspace_limits)
        print('Grasp successful: %r' % (grasp_success))

        color_img, depth_img = self.get_camera_data()
        depth_img = depth_img * self.cam_depth_scale # Apply depth scale from calibration
        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, self.cam_intrinsics, self.cam_pose, self.workspace_limits, self.heightmap_resolution/2)
        #cv2.imwrite("figures/debug.jpg", color_heightmap)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > 0.02] = 1
        empty_threshold = 300
        if self.is_sim and self.is_testing:
            empty_threshold = 10
        if np.sum(stuff_count) < empty_threshold or (self.is_sim and self.no_change_count[0] > 5):
            self.no_change_count = [0, 0]
            if self.is_sim:
                print('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
                terminal = True

        # Detect changes
        depth_diff = abs(depth_heightmap - prev_depth_heightmap)
        depth_diff[np.isnan(depth_diff)] = 0
        depth_diff[depth_diff > 0.3] = 0
        depth_diff[depth_diff < 0.01] = 0
        depth_diff[depth_diff > 0] = 1
        change_threshold = 800
        change_value = np.sum(depth_diff)
        change_detected = change_value > change_threshold or grasp_success
        print('Change detected: %r (value: %d)' % (change_detected, change_value))

        if change_detected:
            self.no_change_count[0] = 0
        else:
            self.no_change_count[0] += 1

        reward = 0.0


        if grasp_success:
            reward = 15.0
        else:
            if change_detected:
                reward = 5.0
        state = [color_heightmap, valid_depth_heightmap, depth_heightmap]
        #print("color heightmap shape",color_heightmap.shape)
        return state, reward, terminal

    def reset(self):
        print("reset simulation")
        self.restart_sim()
        self.add_objects()
        color_img, depth_img = self.get_camera_data()
        depth_img = depth_img * self.cam_depth_scale # Apply depth scale from calibration

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, self.cam_intrinsics, self.cam_pose, self.workspace_limits, self.heightmap_resolution/2)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
        state = [color_heightmap, valid_depth_heightmap, depth_heightmap]
        return state
