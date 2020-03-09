import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import keras
import keras.backend as K
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras.layers import Input, Dense, Dropout, GRU

from data import SessionData
from data import SessionDataLoader
from data import ModelData


class GRUModel:
	def __init__(self, hidden_layer_size, item_size, learning_rate=0.001, batch_size=50):
		self.hidden_layer_size = hidden_layer_size
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.item_size = item_size

		self.model = self.create_mode()

	def create_mode(self):
		inputs = Input(batch_shape=(self.batch_size, 1, self.item_size))

		gru, gru_states = GRU(self.hidden_layer_size, stateful=True, return_state=True)(inputs)

		drop2 = Dropout(0.25)(gru)

		#模型预测输出：Matrix(m,n)m为batch_size，n是长度为item总数的数组，数组中存放每个item的probability
		predictions = Dense(units=self.item_size, activation='softmax')(drop2)
		model = Model(input=inputs, output=[predictions])
		optimizer = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999,
										  epsilon=None, decay=0.0, amsgrad=False)
		model.compile(optimizer=optimizer, loss=categorical_crossentropy)
		model.summary()
		return model

	def get_states(self):
		return [K.get_value(s) for s, _ in self.model.state_updates]

	def evaluate_model(self, test_session_data,recall_k=20,mrr_k=20):
		session_data_loader = SessionDataLoader(test_session_data, batch_size=self.batch_size)

		recall = 0 #召回
		mrr = 0 #Mean Reciprocal Rank
		n = 0
		with tqdm(total=len(test_session_data.session_idx) + 1) as pbar:
			for input,target,mask in session_data_loader:
				target = to_categorical(target,num_classes=self.item_size)
				input = to_categorical(input,num_classes=self.item_size)
				input = np.expand_dims(input,axis=1)

				predict = self.model.predict(input,batch_size=self.batch_size)

				#预测结果的维度： (batch_size序列长度,item_size所有item数量)
				for predict_idx in range(predict.shape[0]):
					n += 1
					predict_row = predict[predict_idx]
					true_row = target[predict_idx]
					recall_idx = predict_row.argsort()[-recall_k:][::-1] #预测itemp的probability的前k个
					mrr_idx = predict_row.argsort()[-mrr_k:][::-1]
					true_idx = true_row.argsort()[-1:]

					if true_idx[0] in recall_idx:
						recall += 1

					if true_idx[0] in mrr_idx:
						mrr += 1 / int((np.where(mrr_idx == true_idx[0]))[0] + 1) #按预测排名计算MRR

					pbar.set_description('Evaluating Model')
					pbar.update(session_data_loader.done_sessions_count)

		recall = recall / n
		mrr = mrr / n
		return (recall,recall_k),(mrr,mrr_k)

	def train_model(self, train_session_data):
		session_data_loader = SessionDataLoader(train_session_data, batch_size=self.batch_size)
		for epoch in range(1,10):
			with tqdm(total=len(train_session_data.session_idx) + 1) as pbar:
				for input, target, mask in session_data_loader:
					# setset hidden stats
					#mask 用来记录已经训练完成的session，训练完成的session的hidden state设置为0
					hidden_states = self.get_states()[0]
					mask_eles = np.ones((self.batch_size, 1))
					mask_eles[mask] = 0
					hidden_states = np.multiply(mask_eles, hidden_states)
					hidden_states = np.array(hidden_states, dtype=np.float32)
					self.model.layers[1].reset_states(hidden_states)

					input_oh = to_categorical(input, num_classes=self.item_size)
					input_oh = np.expand_dims(input_oh, axis=1)
					output_oh = to_categorical(target, num_classes=self.item_size)

					loss = self.model.train_on_batch(input_oh, output_oh)

					pbar.set_description('Epoch{0} loss is {1:.5f}'.format(epoch,loss))
					pbar.update(session_data_loader.done_sessions_count)



if __name__ == '__main__':
	model_data = ModelData('../preprocess/data/ratings.csv')
	train_session_data = SessionData(model_data.tran_data)
	gru_model = GRUModel(hidden_layer_size=100, item_size=len(train_session_data.item_id2index_map),
						 batch_size=50)
	gru_model.train_model(train_session_data)
	test_session_data = SessionData(model_data.test_data)
	print(gru_model.evaluate_model(test_session_data))
