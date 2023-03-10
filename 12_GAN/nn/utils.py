import numpy as np
import cv2

# def plot_image(img, aspect_ratio=1.0, border=1, border_color=0):
# 	'''
# 	Input:
# 		img: with shape (batch_size, 28, 28)
# 	'''
# 	N = img.shape[0]
# 	img_shape = img.shape
# 	img_aspect_ratio = img.shape[1] /  float(img.shape[2])
# 	aspect_ratio *= img_aspect_ratio
# 	tile_height = int(np.ceil(np.sqrt(N * aspect_ratio)))
# 	tile_width = int(np.ceil(np.sqrt(N / aspect_ratio)))
# 	grid_shape = np.zeros((tile_height, tile_width))
#
# 	tile_img



def plot(image):
	'''
	Input:
		img: with shape (batch_size, 28, 28)
	'''
	batch_size = image.shape[0]
	height = image.shape[1]
	width = image.shape[2]
	new_image = np.zeros((batch_idx, 3, height, width))
	for i in range(batch_idx):
		new_image[i] = image[i].astype(np.unit8)
		new_image_red, new_image_green, new_image_blue = new_image





#Created an image (really an ndarray) with three channels
new_image = np.ndarray((3, num_rows, num_cols), dtype=int)

#Did manipulations for my project where my array values went way over 255
#Eventually returned numbers to between 0 and 255

#Converted the datatype to np.uint8
new_image = new_image.astype(np.uint8)

#Separated the channels in my new image
new_image_red, new_image_green, new_image_blue = new_image

#Stacked the channels
new_rgb = np.dstack([new_image_red, new_image_green, new_image_blue])

#Displayed the image
cv2.imshow("WindowNameHere", new_rgbrgb)
cv2.waitKey(0)
