'''
Global settings
'''
import tensorflow as tf


# Default boxes
# DEFAULT_BOXES = ((x1_offset, y1_offset, x2_offset, y2_offset), (...), ...)
# Offset is relative to upper-left-corner and lower-right-corner of the feature map cell
DEFAULT_BOXES = ((-0.5, -0.5, 0.5, 0.5), (0.2, 0.2, -0.2, -0.2), (-0.8, -0.2, 0.8, 0.2), (-0.2, -0.8, 0.2, 0.8))
NUM_DEFAULT_BOXES = len(DEFAULT_BOXES)

# Data chunk size, b/c the raw data cannot be dumped to pickle file all at once
DATA_CHUNK_SIZE = 6000

# Constants
NUM_CLASSES = 2  # 1 objects + 1 background class
NUM_CHANNELS = 1  # grayscale->1, RGB->3
NUM_PRED_CONF = NUM_DEFAULT_BOXES * NUM_CLASSES  # number of class predictions per feature map cell
NUM_PRED_LOC  = NUM_DEFAULT_BOXES * 4  # number of localization regression predictions per feature map cell

# Bounding box parameters
IOU_THRESH = 0.5  # match ground-truth box to default boxes exceeding this IOU threshold, during data prep
NMS_IOU_THRESH = 0.2  # IOU threshold for non-max suppression

# Negatives-to-positives ratio used to filter training data
NEG_POS_RATIO = 5  # negative:positive = NEG_POS_RATIO:1

# Class confidence threshold to count as detection
CONF_THRESH = 0.95

# Model selection and dependent parameters
MODEL = 'AlexNet'  # AlexNet/VGG16/ResNet50
if MODEL == 'AlexNet':
	#IMG_H, IMG_W = 300, 300
	#FM_SIZES = [[36, 36], [17, 17], [9, 9], [5, 5]]  # feature map sizes for SSD hooks via TensorBoard visualization (HxW)

	IMG_H, IMG_W = 250, 400
	FM_SIZES = [[29, 48], [14, 23], [7, 12], [4, 6]]
else:
	raise NotImplementedError('Model not implemented')

# Model hyper-parameters
OPT = tf.train.AdadeltaOptimizer()
REG_SCALE = 1e-2  # L2 regularization strength
LOC_LOSS_WEIGHT = 1.  # weight of localization loss: loss = conf_loss + LOC_LOSS_WEIGHT * loc_loss

# Training process
RESUME = True  # resume training from previously saved model?
NUM_EPOCH = 10
BATCH_SIZE = 32  # batch size for training (relatively small)
VALIDATION_SIZE = 0.05  # fraction of total training set to use as validation set
SAVE_MODEL = True  # save trained model to disk?
SAVE_MODEL_EVERY_EPOCH = True  # save model every epoch?
MODEL_SAVE_PATH = './model.ckpt'  # where to save trained model
