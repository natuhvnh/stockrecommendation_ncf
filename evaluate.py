import pandas as pd 
import numpy as np
import heapq 
from keras.models import load_model
import copy

from utils import *

model = None
topK = None

def evaluate_model(training_model, top_match):
  global model
  global account_dict
  global share_dict
  global model
  global topK
  model = training_model
  topK = top_match
  # load data
  test = pd.read_pickle('data/test.pickle')
  # test = pd.read_pickle('movie_len/test.pickle')
  account_dict = load_dict_from_pickle('data/account_dict.pickle')
  share_dict = load_dict_from_pickle('data/share_dict.pickle')
  # get result
  test['result'] = test.apply(get_result, axis=1)
  # 
  hit_ratio_topK = test['result'].sum(axis=0)/test.shape[0]
  return hit_ratio_topK
def get_result(row):
  map_item_score = {}
  #
  users = copy.copy(row['main_account'])
  users = account_dict[users]
  items = copy.copy(row['test_sample'])
  positive_item = copy.copy(row['ShareCode'])
  positive_item = share_dict[positive_item]
  #
  for i in range(len(items)):
    items[i] = share_dict[items[i]]
  #
  users = np.full(len(items), users, dtype="int32")
  #
  predictions = model.predict([users, np.array(items)], verbose=0)
  #
  for i in range(len(items)):
    item = items[i]
    map_item_score[item] = predictions[i]
  ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
  hit_ratio = get_hit_ratio(ranklist, positive_item)
  return hit_ratio

def get_hit_ratio(ranklist, positive_item):
  for item in ranklist:
    if item == positive_item:
      return 1
  return 0


