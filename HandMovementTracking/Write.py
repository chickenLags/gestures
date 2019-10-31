import numpy as np
import cv2
import argparse
from collections import deque
from matplotlib import pyplot as plt
import json

cap = cv2.VideoCapture(0)
pts = deque(maxlen=64)
img_hue_dict = {}
img_sat_dict = {}
img_val_dict = {}

master_dicts_array = [img_hue_dict, img_sat_dict, img_val_dict]


def add_to_dicts(hsv):
	hue = hsv[0]
	saturation = hsv[1]
	value = hsv[2]

	for key in range(0, 3):
		if hsv[key] in master_dicts_array[key].keys():
			master_dicts_array[key][hsv[key]] += 1
		else:
			master_dicts_array[key][hsv[key]] = 1

def plot_data():
	plt.figure(figsize=(15, 5))

	# for current_dict in master_dicts_array:
	current_dict = master_dicts_array[1]
	plt.subplot(131)
	plt.bar(current_dict.keys(), current_dict.values() )
		
	plt.show()


def get_segments(img):
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	for line in img_hsv:
		for hsv in line:
			add_to_dicts(hsv)

	plot_data()

def maximize_saturation(hsv):
	(h, s, v) = cv2.split(hsv)
	s = np.clip(s, 254, 255)
	hsv = cv2.merge([h, s, v])
	hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	return hsv

def get_edges_image(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	max_sat_hsv_img = maximize_saturation(hsv)
	edges_img = cv2.Canny(max_sat_hsv_img, 180, 200)
	return edges_img

def detect_hands(edges_img):
	pass

def main_loop():
	Lower_green = np.array([10,50,50])
	Upper_green = np.array([250,255,255])
	while True:
		ret, img = cap.read()
		edges_img = get_edges_image(img)
		bounding_boxes = detect_hands(edges_img)

		cv2.imshow("Frame", img)
		cv2.imshow("edges",edges_img)
		# cv2.imshow("hsv", max_sat_hsv_img)
		
		
		k=cv2.waitKey(30) & 0xFF
		if k==32:
			break


def clean_before_close():
	cap.release()
	cv2.destroyAllWindows()


ret, img = cap.read()
if ret:
	# color_segments = get_segments(img)
	main_loop()
clean_before_close()