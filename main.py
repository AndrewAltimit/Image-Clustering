import numpy as np
import cv2
import sys, os
from matplotlib import pyplot as plt
import random

# Given a directory name and an extension of files to search for,
# the function will return a sorted list of files in the folder.
def get_image_paths_from_folder(dir_name, extension):
	# Store current working directory, then change to desired directory
	cwd = os.getcwd()
	os.chdir(dir_name)
	
	# Get the image paths in the folder with the requested extension
	img_list = os.listdir('./')
	img_list = [dir_name + "/" + name for name in img_list if extension.lower() in name.lower() ] 
	img_list.sort()
	
	# Restore the working directory
	os.chdir(cwd)
	
	return img_list


if __name__ == "__main__":

	# Check if all proper input arguments exist
	if len(sys.argv) != 3:
		print("Improper number of input arguments")
		print("USAGE: main.py <image_dir> <number of clusters>")
		sys.exit()
		
	images = get_image_paths_from_folder(sys.argv[1], ".JPEG")
	K = int(sys.argv[2])
	
	for image_path in images:
		# Read in image
		img = cv2.imread(image_path)
		
		# Scale the color channel values to have the same upper bound as the shape upper bound
		scale = max(img.shape[:2]) / 255

		# Extract Indices
		X, Y = np.indices(img.shape[:2])
		X = X.reshape((-1,1))
		Y = Y.reshape((-1,1))

		# Extract B, G, R Color Channels
		B = img[:,:,0].reshape((-1,1)) * scale
		G = img[:,:,1].reshape((-1,1)) * scale
		R = img[:,:,2].reshape((-1,1)) * scale
		
		# 2D Convolution (Image Filtering): https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
		size = 3
		kernel = np.ones((size, size),np.float32)/(size ** 2)
		smoothed_img = cv2.filter2D(img,-1,kernel)
		BS = smoothed_img[:,:,0].reshape((-1,1)) * scale
		GS = smoothed_img[:,:,1].reshape((-1,1)) * scale
		RS = smoothed_img[:,:,2].reshape((-1,1)) * scale
		
		# Concatenate all the features into one dataset
		data = np.float32(np.concatenate((X, Y, B, G, R, BS, GS, RS), axis = 1))
		
		# Specify the termination criteria, number of interations, and upper bound on the amount of change in the center
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		
		# Apply kmeans with the given dataset and criteria definition
		ret, labels, center=cv2.kmeans(data,K,None,criteria,1,cv2.KMEANS_RANDOM_CENTERS)

		# Color in resulting image via the label
		labels = labels.reshape(img.shape[:2])
		for i in range(K):
			img[np.where(labels == i)] = (random.randint(50,255),random.randint(50,255), random.randint(50,255))
		
		# Determine output image path and write output image
		index = image_path.rfind('.')
		result_filename = image_path[:index] + '_K-{}'.format(K) + image_path[index:]
		cv2.imwrite(result_filename, img)