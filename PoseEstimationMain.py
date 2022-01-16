#Vaibhav Bisht
#MS, Mechanical Engineering, Cornell University
#Script for running 'PoseEstimation.py' with input of choice
#Run in directory of FAIR's Detectron2

from PoseEstimation import *
import os
import glob

detector = Detector()

detector.onImage("KEYPOINTS/upright.jpg")