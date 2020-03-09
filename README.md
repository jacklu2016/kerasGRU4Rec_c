# kerasGRU4Rec_c

Just rewrite and add some comment for kerasGRU4Rec, The Original github repository is [kerasGRU4Rec](https://github.com/pcerdam/KerasGRU4Rec)
<br>
This repository offers an implementation of the "Session-based Recommendations With Recurrent Neural Networks" paper (https://arxiv.org/abs/1511.06939) using the Keras framework, tested with TensorFlow backend.
<br>
A script that interprets the MovieLens 20M dataset as if each user's history were one anonymous session (spanning anywhere from months to years) is included. Our implementation presents comparable results to those obtained by the original Theano implementation offered by the GRU4Rec authors
---
# Run model
Dataset:pls download dataset [ml-25m](http://grouplens.org/datasets/),download and unzip it to folder preprocess/data<br>
then run model.py<br>
---
# Requirements
The code has been tested with Python 3.6.8, using the following versions of the required dependencies:<br>

numpy == 1.16.3 <br>
pandas == 0.24.2<br>
tqdm == 4.32.<br>
tensorflow == 1.11.0rc1<br>
keras == 2.2.4<br>
