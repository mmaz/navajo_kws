# %%
import numpy as np
import tensorflow as tf
import os

import sys
sys.path.insert(0, "/home/mark/tinyspeech_harvard/tensorflow/tensorflow/examples/speech_commands/")

import models
import input_data_fix_bg as input_data

# %%
tf.__version__

# %%
WANTED_WORDS="hello,thanks"
# Calculate the percentage of 'silence' and 'unknown' training samples required
# to ensure that we have equal number of samples for each label.
number_of_labels = WANTED_WORDS.count(',') + 1
number_of_total_labels = number_of_labels + 2 # for 'silence' and 'unknown' label
equal_percentage_of_training_samples = int(100.0/(number_of_total_labels))
SILENT_PERCENTAGE = equal_percentage_of_training_samples
UNKNOWN_PERCENTAGE = equal_percentage_of_training_samples

# Constants used during training only
VERBOSITY = 'DEBUG'
EVAL_STEP_INTERVAL = '1000'
SAVE_STEP_INTERVAL = '1000'

# Constants for training directories and filepaths
# LOGS_DIR = 'logs/'
# TRAIN_DIR = 'train/' # for training checkpoints and other files.

# Constants for inference directories and filepaths
# import os
# MODELS_DIR = 'models'
# if not os.path.exists(MODELS_DIR):
#   os.mkdir(MODELS_DIR)
# MODEL_TF = os.path.join(MODELS_DIR, 'KWS_custom.pb')
# MODEL_TFLITE = os.path.join(MODELS_DIR, 'KWS_custom.tflite')
# FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, 'KWS_custom_float.tflite')
# MODEL_TFLITE_MICRO = os.path.join(MODELS_DIR, 'KWS_custom.cc')
# SAVED_MODEL = os.path.join(MODELS_DIR, 'KWS_custom_saved_model')


# Constants which are shared during training and inference
PREPROCESS = 'micro'
WINDOW_STRIDE = 20

# Constants for Quantization
QUANT_INPUT_MIN = 0.0
QUANT_INPUT_MAX = 26.0
QUANT_INPUT_RANGE = QUANT_INPUT_MAX - QUANT_INPUT_MIN

# Constants for audio process during Quantization and Evaluation
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0

# Use the custom local dataset and set the tes/val/train split
DATA_URL = ''
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10

# %%
DATASET_DIR="/home/mark/tinyspeech_harvard/navajo/wav_data/"
bgdir = DATASET_DIR + "_background_noise_"
# %%
model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(WANTED_WORDS.split(','))),
    SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
    WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS)
audio_processor = input_data.AudioProcessor(
    DATA_URL, DATASET_DIR,
    SILENT_PERCENTAGE, UNKNOWN_PERCENTAGE,
    WANTED_WORDS.split(','), VALIDATION_PERCENTAGE,
    TESTING_PERCENTAGE, model_settings, summaries_dir=None, background_data_dir=bgdir)

# %%    
SAVED_MODEL = "../hello_thanks"
FLOAT_MODEL_TFLITE = "../hello_thanks_float.tflite"
MODEL_TFLITE = "../hello_thanks.tflite"
# %%
os.getcwd()

# %%
os.environ["PATH"] += os.pathsep + "/home/mark/miniconda3/envs/tf115/bin"
# %%
os.environ["PATH"]

# %%

REP_DATA_SIZE = 9
with tf.Session() as sess:
  # with tf.compat.v1.Session() as sess: #replaces the above line for use with TF2.x
  float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
  float_tflite_model = float_converter.convert()
  float_tflite_model_size = open(FLOAT_MODEL_TFLITE, "wb").write(float_tflite_model)
  print("Float model is %d bytes" % float_tflite_model_size)

  converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.inference_input_type = tf.lite.constants.INT8
  # converter.inference_input_type = tf.compat.v1.lite.constants.INT8 #replaces the above line for use with TF2.x   
  converter.inference_output_type = tf.lite.constants.INT8
  # converter.inference_output_type = tf.compat.v1.lite.constants.INT8 #replaces the above line for use with TF2.x
  def representative_dataset_gen():
    for i in range(REP_DATA_SIZE):
      data, _ = audio_processor.get_data(1, i*1, model_settings,
                                         BACKGROUND_FREQUENCY, 
                                         BACKGROUND_VOLUME_RANGE,
                                         TIME_SHIFT_MS,
                                         'testing',
                                         sess)
      flattened_data = np.array(data.flatten(), dtype=np.float32).reshape(1, 1960)
      print(i)
      yield [flattened_data]
  converter.representative_dataset = representative_dataset_gen
  tflite_model = converter.convert()
  tflite_model_size = open(MODEL_TFLITE, "wb").write(tflite_model)
  print("Quantized model is %d bytes" % tflite_model_size)

# %%
# Helper function to run inference
def run_tflite_inference_testSet(tflite_model_path, model_type="Float"):
  #
  # Load test data
  #
  np.random.seed(0) # set random seed for reproducible test results.
  with tf.Session() as sess:
    # with tf.compat.v1.Session() as sess: #replaces the above line for use with TF2.x
    test_data, test_labels = audio_processor.get_data(
        -1, 0, model_settings, BACKGROUND_FREQUENCY, BACKGROUND_VOLUME_RANGE,
        TIME_SHIFT_MS, 'testing', sess)
  test_data = np.expand_dims(test_data, axis=1).astype(np.float32)

  #
  # Initialize the interpreter
  #
  interpreter = tf.lite.Interpreter(tflite_model_path)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]
  
  #
  # For quantized models, manually quantize the input data from float to integer
  #
  if model_type == "Quantized":
    input_scale, input_zero_point = input_details["quantization"]
    test_data = test_data / input_scale + input_zero_point
    test_data = test_data.astype(input_details["dtype"])

  #
  # Evaluate the predictions
  #
  correct_predictions = 0
  for i in range(len(test_data)):
    interpreter.set_tensor(input_details["index"], test_data[i])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    top_prediction = output.argmax()
    correct_predictions += (top_prediction == test_labels[i])

  print('%s model accuracy is %f%% (Number of test samples=%d)' % (
      model_type, (correct_predictions * 100) / len(test_data), len(test_data)))

# %%
# Compute float model accuracy
run_tflite_inference_testSet(FLOAT_MODEL_TFLITE)

# Compute quantized model accuracy
run_tflite_inference_testSet(MODEL_TFLITE, model_type='Quantized')