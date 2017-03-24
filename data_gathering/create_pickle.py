'''
Create raw data pickle file
data_raw is a dict mapping image_filename -> [{'class': class_int, 'box_coords': (x1, y1, x2, y2)}, {...}, ...]
'''
import numpy as np
import pickle
import re
import os
import time
from PIL import Image

# Script config
RESIZE_IMAGE = True  # resize the images and write to 'resized_images/'
GRAYSCALE = True  # convert image to grayscale? this option is only valid if RESIZE_IMAGE==True (FIXME)
TARGET_W, TARGET_H = 400, 250  # original image is 1920x1200, keep 1.6 aspect ratio
DEBUG = False

# Raw data dict and label map
data_raw = {}
label_map = {'car': 1}  # background class is 0

# Keep track of time
t0 = time.time()

######################################################
# Parse Dataset 1 (object-detection-crowdai/*)
######################################################
data_dir = 'object-detection-crowdai'

# For speed, put entire contents of labels.csv in memory
labels_csv = []
with open(data_dir + '/labels.csv', 'r') as f:
	for line in f:
		line = line[:-1]  # strip trailing newline
		labels_csv.append(line)

# Create pickle file to represent dataset
image_files = os.listdir(data_dir)
for count, image_file in enumerate(image_files):
	if DEBUG:
		if count > 100:
			break
	if (count+1) % 100 == 0:
		print('Processed %d images in %s - total elapsed time: %d sec' % (count+1, data_dir, int(time.time() - t0)))
	if image_file == 'labels.csv':
		continue

	new_image_file = 'd1_' + image_file

	# Find box coordinates for all objects in this image
	class_list = []
	box_coords_list = []
	for line in labels_csv:
		if re.search(image_file, line):
			fields = line.split(',')

			# Get label name and assign class label
			label_name = fields[5]
			if label_name != 'Car':
				continue  # ignore certain labels
			label_class = label_map[label_name.lower()]
			class_list.append(label_class)

			# Resize image, get rescaled box coordinates
			box_coords = np.array([int(x) for x in fields[0:4]])

			if RESIZE_IMAGE:
				# Resize the images and write to 'resized_images/'
				image = Image.open(os.path.join(data_dir, image_file))
				orig_w, orig_h = image.size

				if GRAYSCALE:
					image = image.convert('L')  # 8-bit grayscale
				image = image.resize((TARGET_W, TARGET_H), Image.LANCZOS)  # high-quality downsampling filter

				resized_dir = 'resized_images_%dx%d/' % (TARGET_W, TARGET_H)
				if not os.path.exists(resized_dir):
					os.makedirs(resized_dir)

				image.save(os.path.join(resized_dir, new_image_file))

				# Rescale box coordinates
				x_scale = TARGET_W / orig_w
				y_scale = TARGET_H / orig_h

				ulc_x, ulc_y, lrc_x, lrc_y = box_coords
				new_box_coords = (ulc_x * x_scale, ulc_y * y_scale, lrc_x * x_scale, lrc_y * y_scale)
				new_box_coords = [round(x) for x in new_box_coords]
				box_coords = np.array(new_box_coords)

			box_coords_list.append(box_coords)

	if len(class_list) == 0:
		continue  # ignore images with no labels-of-interest
	class_list = np.array(class_list)
	box_coords_list = np.array(box_coords_list)

	# Create the list of dicts
	the_list = []
	for i in range(len(box_coords_list)):
		d = {'class': class_list[i], 'box_coords': box_coords_list[i]}
		the_list.append(d)

	data_raw[new_image_file] = the_list


######################################################
# Parse Dataset 2 (object-dataset/*)
######################################################
data_dir = 'object-dataset'

# For speed, put entire contents of labels.csv in memory
labels_csv = []
with open(data_dir + '/labels.csv', 'r') as f:
	for line in f:
		line = line[:-1]  # strip trailing newline
		labels_csv.append(line)

# Create pickle file to represent dataset
image_files = os.listdir(data_dir)
for count, image_file in enumerate(image_files):
	if DEBUG:
		if count > 100:
			break
	if (count+1) % 100 == 0:
		print('Processed %d images in %s - total elapsed time: %d sec' % (count+1, data_dir, int(time.time() - t0)))
	if image_file == 'labels.csv':
		continue

	new_image_file = 'd2_' + image_file

	# Find box coordinates for all objects in this image
	class_list = []
	box_coords_list = []
	for line in labels_csv:
		if re.search(image_file, line):
			fields = line.split(' ')

			# Get label name and assign class label
			label_name = fields[6]
			if label_name != '"car"':
				continue  # ignore certain labels
			label_class = label_map[label_name[1:-1]]  # remove the quotation marks
			class_list.append(label_class)

			# Resize image, get rescaled box coordinates
			box_coords = np.array([int(x) for x in fields[1:5]])

			if RESIZE_IMAGE:
				# Resize the images and write to 'resized_images/'
				image = Image.open(os.path.join(data_dir, image_file))
				orig_w, orig_h = image.size

				if GRAYSCALE:
					image = image.convert('L')  # 8-bit grayscale
				image = image.resize((TARGET_W, TARGET_H), Image.LANCZOS)  # high-quality downsampling filter

				resized_dir = 'resized_images_%dx%d/' % (TARGET_W, TARGET_H)
				if not os.path.exists(resized_dir):
					os.makedirs(resized_dir)

				image.save(os.path.join(resized_dir, new_image_file))

				# Rescale box coordinates
				x_scale = TARGET_W / orig_w
				y_scale = TARGET_H / orig_h

				ulc_x, ulc_y, lrc_x, lrc_y = box_coords
				new_box_coords = (ulc_x * x_scale, ulc_y * y_scale, lrc_x * x_scale, lrc_y * y_scale)
				new_box_coords = [round(x) for x in new_box_coords]
				box_coords = np.array(new_box_coords)

			box_coords_list.append(box_coords)

	if len(class_list) == 0:
		continue  # ignore images with no labels-of-interest
	class_list = np.array(class_list)
	box_coords_list = np.array(box_coords_list)

	# Create the list of dicts
	the_list = []
	for i in range(len(box_coords_list)):
		d = {'class': class_list[i], 'box_coords': box_coords_list[i]}
		the_list.append(d)

	data_raw[new_image_file] = the_list


######################################################
# Save results to pickle file
######################################################
with open('data_raw_%dx%d.p' % (TARGET_W, TARGET_H), 'wb') as f:
	pickle.dump(data_raw, f)
