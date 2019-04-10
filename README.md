# Semantic Image Search

## Introduction
This project extends the original [Google DeViSE](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41473.pdf) paper to create a functioning image search engine with a focus on interpreting search results. We have extended the original paper in the following ways. First, we added an RNN to process variable length queries as opposed to single words. It has been tested on subsets of the [UIUC-PASCAL dataset](http://vision.cs.uiuc.edu/pascal-sentences/) and the final network has been trained on the [MSCOCO 2014 dataset](http://cocodataset.org/#home).


## Development History
**master**: This repository houses code to replicate the base DeViSE paper. Since the project - and it’s scope - has grown organically as we advanced, we decided to branch out and leave the vanilla DeViSE code base intact. The **master** branch contains code to setup the experiment, download and pre-process data for implementing the paper. `model.py` contains the code for the model. Due to computational constraints, the experiments are run on the UIUC-PASCAL sentences dataset as opposed to ImageNet. This dataset contains 50 images per category for 20 categories along with 5 captions per image. 

## How to run
Run the server using: 

```
python server_lime_contours.py \
--word_index=/path/to/devise_cache/DICT_word_index.VAL.pkl \
--cache=/path/to/devise_cache/cache.h5 \
--model=/path/to/devise_cache/epoch_9.hdf5 \
--threaded=0 \
--host=127.0.0.1 \
--port=5000 \
--dummy=0 \
--captions_train=/path/to/devise_cache/annotations/captions_train2014.json \
--captions_valid=/path/to/devise_cache/annotations/captions_val2014.json \
--vgg16=/path/to/devise_cache/vgg16_weights_th_dim_ordering_th_kernels.h5
```

 Open a modern web browser and navigate to localhost:8000 to view the webpage.

## How it works

### DeViSE: A Deep Visual-Semantic Embedding Model
 In this paper, the authors present a new deep visual-semantic embedding model trained to identify visual objects using both labeled image data as well as semantic information gleaned from unannotated text. They accomplish this by minimizing a combination of the cosine similarity and hinge rank loss between the embedding vectors learned by the language model and the vectors from the core visual model as shown below. 
 
 ![devise_core](https://user-images.githubusercontent.com/8658591/34649979-1b47f090-f3df-11e7-8833-e488dc33cad0.PNG)

In the interest of time, we did not train a skip-gram model ourselves but chose to use the [GloVe (Global Vectors for Word Representation)](https://nlp.stanford.edu/projects/glove/) model from the stanford NLP group as our initilization. 
