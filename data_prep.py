'''
Data preparation
'''
from settings import *
import numpy as np
import pickle
import time
import math


def calc_iou(box_a, box_b):
	"""
	Calculate the Intersection Over Union of two boxes
	Each box specified by upper left corner and lower right corner:
	(x1, y1, x2, y2), where 1 denotes upper left corner, 2 denotes lower right corner

	Returns IOU value
	"""
	# Calculate intersection, i.e. area of overlap between the 2 boxes (could be 0)
	# http://math.stackexchange.com/a/99576
	x_overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
	y_overlap = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
	intersection = x_overlap * y_overlap

	# Calculate union
	area_box_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
	area_box_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
	union = area_box_a + area_box_b - intersection

	iou = intersection / union
	return iou


def find_gt_boxes(data_raw, image_file):
	"""
	Given (global) feature map sizes, and single training example,
	find all default boxes that exceed Jaccard overlap threshold

	Returns y_true array that flags the matching default boxes with class ID (-1 means nothing there)
	"""
	# Pre-process ground-truth data
	# Convert absolute coordinates to relative coordinates ranging from 0 to 1
	# Read the object class label (note background class label is 0, object labels are ints >=1)
	objects_data = data_raw[image_file]

	objects_class = []
	objects_box_coords = []  # relative coordinates
	for object_data in objects_data:
		# Find class label
		object_class = object_data['class']
		objects_class.append(object_class)

		# Calculate relative coordinates
		# (x1, y1, x2, y2), where 1 denotes upper left corner, 2 denotes lower right corner
		abs_box_coords = object_data['box_coords']
		scale = np.array([IMG_W, IMG_H, IMG_W, IMG_H])
		box_coords = np.array(abs_box_coords) / scale
		objects_box_coords.append(box_coords)

	# Initialize y_true to all 0s (0 -> background)
	y_true_len = 0
	for fm_size in FM_SIZES:
		y_true_len += fm_size[0] * fm_size[1] * NUM_DEFAULT_BOXES
	y_true_conf = np.zeros(y_true_len)
	y_true_loc = np.zeros(y_true_len * 4)

	# For each GT box, for each feature map, for each feature map cell, for each default box:
	# 1) Calculate the Jaccard overlap (IOU) and annotate the class label
	# 2) Count how many box matches we got
	# 3) If we got a match, calculate normalized box coordinates and updte y_true_loc
	match_counter = 0
	for i, gt_box_coords in enumerate(objects_box_coords):
		y_true_idx = 0
		for fm_size in FM_SIZES:
			fm_h, fm_w = fm_size  # feature map height and width
			for row in range(fm_h):
				for col in range(fm_w):
					for db in DEFAULT_BOXES:
						# Calculate relative box coordinates for this default box
						x1_offset, y1_offset, x2_offset, y2_offset = db
						abs_db_box_coords = np.array([
							max(0, col + x1_offset),
							max(0, row + y1_offset),
							min(fm_w, col+1 + x2_offset),
							min(fm_h, row+1 + y2_offset)
						])
						scale = np.array([fm_w, fm_h, fm_w, fm_h])
						db_box_coords = abs_db_box_coords / scale

						# Calculate Jaccard overlap (i.e. Intersection Over Union, IOU) of GT box and default box
						iou = calc_iou(gt_box_coords, db_box_coords)

						# If box matches, i.e. IOU threshold met
						if iou >= IOU_THRESH:
							# Update y_true_conf to reflect we found a match, and increment match_counter
							y_true_conf[y_true_idx] = objects_class[i]
							match_counter += 1

							# Calculate normalized box coordinates and update y_true_loc
							abs_box_center = np.array([col + 0.5, row + 0.5])  # absolute coordinates of center of feature map cell
							abs_gt_box_coords = gt_box_coords * scale  # absolute ground truth box coordinates (in feature map grid)
							norm_box_coords = abs_gt_box_coords - np.concatenate((abs_box_center, abs_box_center))
							y_true_loc[y_true_idx*4 : y_true_idx*4 + 4] = norm_box_coords

						y_true_idx += 1

	return y_true_conf, y_true_loc, match_counter


def do_data_prep(data_raw, chunk_size):
	"""
	Create the y_true array
	data_raw is the dict mapping image_file -> [{'class': class_int, 'box_coords': (x1, y1, x2, y2)}, {...}, ...]

	Creates a dict {image_file1: {'y_true_conf': y_true_conf, 'y_true_loc': y_true_loc}, image_file2: ...}

	Dump partial data to data_prep_*__x.p every chunk_size images
	"""
	# Keep track of total number of images with matchin boxes
	total_match = 0

	# Prepare the data by populating y_true appropriately
	t0 = time.time()  # keep track of time elapsed
	data_prep = {}
	for count, image_file in enumerate(data_raw.keys()):
		if (count+1) % 100 == 0:
			print('Processed %d images, %d good ones - total elapsed time: %d sec' % (count+1, len(data_prep.keys()), int(time.time() - t0)))

		# Find groud-truth boxes based on Jaccard overlap,
		# populate y_true_conf (class labels) and y_true_loc (normalized box coordinates)
		y_true_conf, y_true_loc, match_counter = find_gt_boxes(data_raw, image_file)

		# Only want data points where we have matching default boxes -- FIXME: Nope, keep them all
		#if match_counter > 0:
		data_prep[image_file] = {'y_true_conf': y_true_conf, 'y_true_loc': y_true_loc}

		# Dump partial data to pickle file if applicable
		if (count+1) % DATA_CHUNK_SIZE == 0:
			chunk_idx = (count+1) // DATA_CHUNK_SIZE

			print('Dumping partial data for chunk %d' % chunk_idx)
			print('Images with >=1 matching box: %d' % len(data_prep.keys()))
			total_match += len(data_prep.keys())

			with open('data_prep_%sx%s__%d.p' % (IMG_W, IMG_H, chunk_idx), 'wb') as f:
				pickle.dump(data_prep, f)
			data_prep = {}

	# Dump the last few data points, if applicable
	total_data_length = len(data_raw.keys())
	if not total_data_length % DATA_CHUNK_SIZE == 0:
		chunk_idx = math.ceil(total_data_length / DATA_CHUNK_SIZE)
		print('Dumping partial data for chunk %d' % chunk_idx)
		with open('data_prep_%sx%s__%d.p' % (IMG_W, IMG_H, chunk_idx), 'wb') as f:
			pickle.dump(data_prep, f)

	return total_match


if __name__ == '__main__':
	with open('data_raw_%sx%s.p' % (IMG_W, IMG_H), 'rb') as f:
		data_raw = pickle.load(f)

	print('Preparing data (i.e. matching boxes)')
	total_match = do_data_prep(data_raw, DATA_CHUNK_SIZE)

	print('Total images with >=1 matching box: %d' % total_match)
