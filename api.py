import pandas as pd 
import numpy as np
import pickle
from flask import Flask, request, jsonify
import tensorflow as tf
from utils import load_dict_from_pickle

app = Flask(__name__)
#
@app.route('/api',methods=['POST'])
def get_topK_recommend_stock():
  topK=10
  # get data from post request
  data = request.get_json(force=True)
  main_account = data['main_account']
  main_account_encode = account_dict[main_account]
  user_stocks_prediction = {}
  for j in share_dict:
    stock_encode = share_dict[j]
    if (main_account_encode, stock_encode) not in interaction_matrix:
      account = np.array([main_account_encode])
      stock = np.array([stock_encode])
      # need to predict in 
      predicted_value = model.predict([account, stock])   # [[value]]
      #
      predicted_value = predicted_value.item()  # get value
      user_stocks_prediction[j] = predicted_value
  user_stocks_prediction = sorted(user_stocks_prediction.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
  recommend_stocks = []
  for i in range(topK):
    stock = user_stocks_prediction[i][0]
    recommend_stocks.append(stock)
  return jsonify(recommend_stocks)

if __name__ == '__main__':
  # load data
  model = pickle.load(open('model/NCF_8_[8, 8, 8, 8]_0.6663_1_0.282744.pkl', 'rb'))
  interaction_matrix = load_dict_from_pickle('data/interaction_matrix.pickle')
  account_dict = load_dict_from_pickle('data/account_dict.pickle')
  share_dict = load_dict_from_pickle('data/share_dict.pickle')
  #
  app.run(port=5000, debug=True, use_reloader=False)  # `use_reloader=False` = flask not reload twice, but not reload when changes

