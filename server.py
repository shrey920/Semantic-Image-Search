import time
from flask import render_template, jsonify, request
from multiprocessing import Lock
from keras.models import load_model
from keras.layers import Lambda
import keras.backend as K
from keras.preprocessing.text import text_to_word_sequence
import os, sys
import pickle
import h5py
# from rnn_model import hinge_rank_loss
import ipdb
import numpy as np
from flask import Flask
import tensorflow as tf
import random
import argparse
import urllib
import cStringIO
from PIL import Image
import cv2
import sqlite3 

parser = argparse.ArgumentParser(description='server')
parser.add_argument("--word_index", type=str, help="location of the DICT_word_index.VAL/TRAIN.pkl", required=True)
parser.add_argument("--cache", type=str, help="location of the cache.h5 file", required=True)
parser.add_argument("--model", type=str, help="location of the model.hdf5 snapshot", required=True)
parser.add_argument("--threaded", type=int, help="Run flask server in multi--threaded/single--threaded mode", required=True)
parser.add_argument("--host", type=str, help="flask server host in app.run()", required=True)
parser.add_argument("--port", type=int, help="port on which the server will be run", required=True)
parser.add_argument("--dummy", type=int, help="run server in dummy mode for testing js/html/css etc.", required=True)
parser.add_argument("--captions_train", type=str, help="location of string captions of training images", required=True)
parser.add_argument("--captions_valid", type=str, help="location of string captions of validation images", required=True)
parser.add_argument("--vgg16", type=str, help="location of vgg16 weights", required=True)
args = parser.parse_args()

app = Flask(__name__)

DUMMY_MODE = bool(args.dummy)
MODEL_LOC = args.model
WORD_DIM = 300

# VERY IMPORTANT VARIABLES
mutex = Lock()
MAX_SEQUENCE_LENGTH = 20
MODEL=None
DICT_word_index = None

MARGIN = 0.2
INCORRECT_BATCH = 32
BATCH = INCORRECT_BATCH + 1

# Load Spacy 
from nlp_stuff import QueryParser
QPObj = QueryParser()

def hinge_rank_loss(y_true, y_pred, TESTING=False):
	"""
	Custom hinge loss per (image, label) example - Page4.
	
	Keras mandates the function signature to follow (y_true, y_pred)
	In devise:master model.py, this function accepts:
	- y_true as word_vectors
	- y_pred as image_vectors

	For the rnn_model, the image_vectors and the caption_vectors are concatenated.
	This is due to checks that Keras has enforced on (input,target) sizes 
	and the inability to handle multiple outputs in a single loss function.

	These are the actual inputs to this function:
	- y_true is just a dummy placeholder of zeros (matching size check)
	- y_pred is concatenate([image_output, caption_output], axis=-1)
	The image, caption features are first separated and then used.
	"""
	## y_true will be zeros
	select_images = lambda x: x[:, :WORD_DIM]
	select_words = lambda x: x[:, WORD_DIM:]

	slice_first = lambda x: x[0:1 , :]
	slice_but_first = lambda x: x[1:, :]

	# separate the images from the captions==words
	image_vectors = Lambda(select_images, output_shape=(BATCH, WORD_DIM))(y_pred)
	word_vectors = Lambda(select_words, output_shape=(BATCH, WORD_DIM))(y_pred)

	# separate correct/wrong images
	correct_image = Lambda(slice_first, output_shape=(1, WORD_DIM))(image_vectors)
	wrong_images = Lambda(slice_but_first, output_shape=(INCORRECT_BATCH, WORD_DIM))(image_vectors)

	# separate correct/wrong words
	correct_word = Lambda(slice_first, output_shape=(1, WORD_DIM))(word_vectors)
	wrong_words = Lambda(slice_but_first, output_shape=(INCORRECT_BATCH, WORD_DIM))(word_vectors)

	# l2 norm
	l2 = lambda x: K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
	l2norm = lambda x: x/l2(x)

	# tiling to replicate correct_word and correct_image
	correct_words = K.tile(correct_word, (INCORRECT_BATCH,1))
	correct_images = K.tile(correct_image, (INCORRECT_BATCH,1))

	# converting to unit vectors
	correct_words = l2norm(correct_words)
	wrong_words = l2norm(wrong_words)
	correct_images = l2norm(correct_images)
	wrong_images = l2norm(wrong_images)

	# correct_image VS incorrect_words | Note the singular/plurals
	cost_images = MARGIN - K.sum(correct_images * correct_words, axis=1) + K.sum(correct_images * wrong_words, axis=1) 
	cost_images = K.maximum(cost_images, 0.0)
	
	# correct_word VS incorrect_images | Note the singular/plurals
	cost_words = MARGIN - K.sum(correct_words * correct_images, axis=1) + K.sum(correct_words * wrong_images, axis=1) 
	cost_words = K.maximum(cost_words, 0.0)

	# currently cost_words and cost_images are vectors - need to convert to scalar
	cost_images = K.sum(cost_images, axis=-1)
	cost_words  = K.sum(cost_words, axis=-1)

	if TESTING:
		# ipdb.set_trace()
		assert K.eval(wrong_words).shape[0] == INCORRECT_BATCH
		assert K.eval(correct_words).shape[0] == INCORRECT_BATCH
		assert K.eval(wrong_images).shape[0] == INCORRECT_BATCH
		assert K.eval(correct_images).shape[0] == INCORRECT_BATCH
		assert K.eval(correct_words).shape==K.eval(correct_images).shape
		assert K.eval(wrong_words).shape==K.eval(wrong_images).shape
		assert K.eval(correct_words).shape==K.eval(wrong_images).shape
	
	return (cost_words + cost_images) / INCORRECT_BATCH


if DUMMY_MODE==False:
	
	MODEL = load_model(MODEL_LOC, custom_objects={"hinge_rank_loss":hinge_rank_loss})
	graph = tf.get_default_graph()
	
	print MODEL.summary()
	
	assert os.path.isfile(args.word_index), "Could not find {}".format(args.word_index)	
	
	with open(args.word_index,"r") as f:
		DICT_word_index = pickle.load(f)
	assert DICT_word_index is not None, "Could not load dictionary that maps word to index"

	im_outs = None 
	fnames = None
	with h5py.File(args.cache) as F:
		im_outs = F["data/im_outs"][:]
		fnames  = F["data/fnames"][:]
	assert im_outs is not None, "Could not load im_outs from cache.h5"
	assert fnames is not None, "Could not load fnames from cache.h5"

	# load the string captions from .json file 
	from pycocotools.coco import COCO
	train_caps = COCO(args.captions_train)
	valid_caps = COCO(args.captions_valid)
	

def coco_url_to_flickr_url(coco_urls):
	'''
	mscoco.org does no longer host the images. Hence we convert the urls from mscoco.org/images/imgid to its flickr url 
	'''
	flickr_urls = []
	for url in coco_urls:
		imgId = int(url.split("/")[-1])
		fl_url = valid_caps.imgs[imgId]["flickr_url"] # Extract the flickr url from valid_caps (not doing from train_caps yet)
		flickr_urls.append(fl_url)

	assert len(flickr_urls) == len(coco_urls), "flickr_urls is not same length as coco_urls"
	return flickr_urls

# Query string -> word index list 
def query_string_to_word_indices(query_string):
	
	# string -> list of words 
	words = text_to_word_sequence(
			text = query_string,
			filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
			lower=True,
			split=" "
		)

	# check if words in dictionary
	all_words = DICT_word_index.keys()
	for word in words:
		if word not in all_words: 
			_err_msg = "could not find word  | {} | in server's dictionary".format(word)
			raise ValueError(_err_msg)

	# list of words -> list of indices
	words_index = []
	for word in words:
		words_index.append(DICT_word_index[word])

	# pad to 20 words
	if len(words_index) < MAX_SEQUENCE_LENGTH:
		padding = [0 for _ in range(MAX_SEQUENCE_LENGTH - len(words_index))]
		words_index += padding

	if len(words_index) != MAX_SEQUENCE_LENGTH:
		raise ValueError("words_index is not {} numbers long".format(MAX_SEQUENCE_LENGTH))

	return np.array(words_index).reshape((1,MAX_SEQUENCE_LENGTH))

@app.route("/")
@app.route("/index")
def index():
	return render_template("index.html", title="Home")

def run_model(query_string):
	''' This fxn takes a query string
	runs it through the Keras model and returns result.'''

	# run forward pass
	# find diff 
	# get images having closest diff
	print "..waiting to acquire lock"
	result = None
	with mutex:
		print "lock acquired, running model..."
		if DUMMY_MODE:
			
			time.sleep(2)
			
			result = ["static/12345.jpg", "static/32561.jpg", "static/45321.jpg"] 
			
			# captions = ["the quick brown fox jumps over the lazy dog."]
			# import copy 
			# captions = copy.deepcopy(captions) + copy.deepcopy(captions) + copy.deepcopy(captions) + copy.deepcopy(captions) + copy.deepcopy(captions) # each image has 5 captions  
			# captions = [ copy.deepcopy(captions) for i in range(3)]                       # we have 3 images, each with 5 captions
			
			# assert len(captions) == len(result), " #results != #captions"

			coco_urls = result 
			flickr_urls = result
						
		else:
			assert MODEL is not None, "not in dummy mode, but model did not load!"

			# convert query string to word_index
			try:
				word_indices = query_string_to_word_indices(query_string)
			except Exception, e:
				print str(e)
				return 2, str(e), [], []

			## multithread fix for keras/tf backend
			global graph
			with graph.as_default():
				# forward pass 
				output = MODEL.predict([ np.zeros((1,4096)) , word_indices ])[:, WORD_DIM: ]
				output = output / np.linalg.norm(output, axis=1, keepdims=True)
			
				# compare with im_outs
				TOP_K = 50
				diff = im_outs - output 
				diff = np.linalg.norm(diff, axis=1)
				top_k_indices = np.argsort(diff)[:TOP_K].tolist()

				# populate "results" with fnames of top_k_indices
				result = []
				for k in top_k_indices:
					result.append(fnames[k][0])

				# Replace /var/coco/train2014_clean/COCO_train2014_000000364251.jpg with http://mscoco.org/images/364251
				coco_urls = []
				for r in result:

					imname = r.split("/")[-1] # COCO_train2014_000000364251.jpg
					imname = imname.split("_")[-1] # 000000364251.jpg
					
					i = 0
					while imname[i] == "0":
						i += 1
					imname = imname[i:] # 364251.jpg
					imname = imname.rstrip(".jpg") # 364251
					imname = "http://mscoco.org/images/" + imname # http://mscoco.org/images/364251

					coco_urls.append(imname)
				
				#### NOTE: Since MSCOCO.ORG NO longer hosts images, we need to fetch images from flickr #####
				flickr_urls = coco_url_to_flickr_url(coco_urls)
				
				
							
		print '..over'
	
	if result is None or len(result)<2:
		return 1,"Err. Model prediction returned None. If you're seeing this, something went horribly wrong at our end.", [], []
	else:
		return 0, flickr_urls, coco_urls

@app.route("/_process_query")
def process_query():

	query_string = request.args.get('query', type=str)
	rc, flickr_urls, coco_urls = run_model(query_string) 
	
	result = {
		"rc":rc,
		"flickr_urls": flickr_urls,
		"coco_urls" : coco_urls
	}

	return jsonify(result)


if __name__ == '__main__':
	app.run(threaded=bool(args.threaded), host=args.host, port=args.port)
