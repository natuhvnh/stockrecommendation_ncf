import pandas as pd 
import numpy as np
from keras.models import load_model
from utils import *

# load data
model = pickle.load(open('model/NCF_8_[8, 8, 8, 8]_0.5788_1_0.093690.pkl', 'rb'))
interaction_matrix = load_dict_from_pickle('data/interaction_matrix.pickle')
account_dict = load_dict_from_pickle('data/account_dict.pickle')
share_dict = load_dict_from_pickle('data/share_dict.pickle')
#
main_account = '0001898ab3584ff6a99ec11ffcb1f09c'
main_account_encode = account_dict[main_account]
topK = 10
#
def get_topK_recommend_stock(main_account, topK):
  user_stocks_prediction = {}
  for j in share_dict:
    stock_encode = share_dict[j]
    if (main_account_encode, stock_encode) not in interaction_matrix:
      account = np.array([main_account_encode])
      stock = np.array([stock_encode])
      predicted_value = model.predict([account, stock])   # [[value]]
      predicted_value = predicted_value.item()  # get value
      user_stocks_prediction[j] = predicted_value
  user_stocks_prediction = sorted(user_stocks_prediction.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
  recommend_stocks = []
  for i in range(topK):
    stock = user_stocks_prediction[i][0]
    recommend_stocks.append(stock)
  return recommend_stocks

recommend_stocks = get_topK_recommend_stock(main_account, topK)
print(recommend_stocks)