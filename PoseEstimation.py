#Vaibhav Bisht
#MS, Mechanical Engineering, Cornell University
#Script for Knee, hip, and ankle angles using keypoints found using Mask R-CNN 
#Run in directory of FAIR's Detectron2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math

import os
import cv2
import numpy as np

class Detector:
	def __init__(self):
		self.cfg = get_cfg()

		#Load model config and pretrained model

		self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
		self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
                
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
		self.cfg.MODEL.DEVICE = "cpu" #cpu or cuda

		self.predictor = DefaultPredictor(self.cfg)

	def onImage(self, imagePath):
		image = cv2.imread(imagePath)

		predictions = self.predictor(image)
		viz = Visualizer(image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
		
		out = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
		result = out.get_image()[:,:,::-1]

		#cv2.imshow("Result", result)
		#cv2.waitKey()
		cv2.imwrite("./KEYPOINTS/upright_keypoints.jpg", out.get_image()[:,:,::-1])

		structure = predictions['instances']
		keypoints = structure.pred_keypoints
		keypoints = np.float32(keypoints)

		#Since keypoints is an array within an array, we extract the array from array
		keys = keypoints[0]

		x = []
		y = []
		z = []
		for i in range(0, len(keys)):
			x.append(keys[i][0])
			y.append(keys[i][1])
			z.append(keys[i][2])

		#3D SCATTER PLOT OF KEYPOINTS
		#plt.rcParams["figure.figsize"] = [7.00, 3.50]
		#plt.rcParams["figure.autolayout"] = True
		#fig = plt.figure()
		#ax = fig.add_subplot(111, projection='3d')
		#ax.scatter3D(x, y, z, c='blue');
		#plt.savefig("test.png")

		#2D SCATTER PLOT OF KEYPOINTS
		#fig = plt.scatter(x, y, c='blue')
		#plt.savefig("testo.png")

		#ALL FOR ANGLE BETWEEN BACK AND THIGH
		hips = []
		hips.append((x[11]+x[12])/2)
		hips.append((y[11]+y[12])/2)
		hips.append((z[11]+z[12])/2)

		shoulder = []
		shoulder.append((x[5]+x[6])/2)
		shoulder.append((y[5]+y[6])/2)
		shoulder.append((z[5]+z[6])/2)

		back = []
		back.append(shoulder[0]-hips[0])
		back.append(shoulder[1]-hips[1])
		back.append(shoulder[2]-hips[2])

		knee1_back = []
		knee1_back.append(x[13]-x[11])
		knee1_back.append(y[13]-y[11])
		knee1_back.append(z[13]-z[11])

		knee2_back = []
		knee2_back.append(x[14]-x[12])
		knee2_back.append(y[14]-y[12])
		knee2_back.append(z[14]-z[12])

		angle_knee1_back = (back[0]*knee1_back[0] + back[1]*knee1_back[1] + back[2]*knee1_back[2]) / (math.sqrt((back[0]**2) + (back[1]**2) + (back[2]**2)) * math.sqrt((knee1_back[0]**2) + (knee1_back[1]**2) + (knee1_back[2]**2)))
		angle_knee1_back = (math.acos(angle_knee1_back)*180)/math.pi
		print('Angle between first knee and back', angle_knee1_back)

		angle_knee2_back = (back[0]*knee2_back[0] + back[1]*knee2_back[1] + back[2]*knee2_back[2]) / (math.sqrt((back[0]**2) + (back[1]**2) + (back[2]**2)) * math.sqrt((knee2_back[0]**2) + (knee2_back[1]**2) + (knee2_back[2]**2)))
		angle_knee2_back = (math.acos(angle_knee2_back)*180)/math.pi
		print('Angle between second knee and back', angle_knee2_back)

		#ALL FOR ANGLE BETWEEN THIGH AND SHIN
		knee1_shin = []
		knee1_shin.append(x[11]-x[13]); knee1_shin.append(y[11]-y[13]); knee1_shin.append(z[11]-z[13])

		knee2_shin = []
		knee2_shin.append(x[12]-x[14]); knee2_shin.append(y[12]-y[14]); knee2_shin.append(z[12]-z[14])

		shin1 = []
		shin1.append(x[15]-x[13]); shin1.append(y[15]-y[13]); shin1.append(z[15]-z[13])

		shin2 = []
		shin2.append(x[16]-x[14]); shin2.append(y[16]-y[14]); shin2.append(z[16]-z[14])

		angle_knee1_shin = (shin1[0]*knee1_shin[0] + shin1[1]*knee1_shin[1] + shin1[2]*knee1_shin[2]) / (math.sqrt((shin1[0]**2) + (shin1[1]**2) + (shin1[2]**2)) * math.sqrt((knee1_shin[0]**2) + (knee1_shin[1]**2) + (knee1_shin[2]**2)))
		angle_knee1_shin = (math.acos(angle_knee1_shin)*180)/math.pi
		print('Angle between first knee and shin', angle_knee1_shin)

		angle_knee2_shin = (shin2[0]*knee2_shin[0] + shin2[1]*knee2_shin[1] + shin2[2]*knee2_shin[2]) / (math.sqrt((shin2[0]**2) + (shin2[1]**2) + (shin2[2]**2)) * math.sqrt((knee2_shin[0]**2) + (knee2_shin[1]**2) + (knee2_shin[2]**2)))
		angle_knee2_shin = (math.acos(angle_knee2_shin)*180)/math.pi
		print('Angle between second knee and shin', angle_knee2_shin)
