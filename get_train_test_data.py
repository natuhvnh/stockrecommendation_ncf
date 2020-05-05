import pandas as pd
import numpy as np
import random
from utils import *

# Variable
time = "t9_2019"
min_stock_hold = 5
random_test_size = 100
# Get data
stock = get_df_from_mysql(database_name="Stock", table_name=time)
stock = split_type_and_main_account(stock, "AccountCode")
stock = stock.groupby(['main_account', 'ShareCode'])['ShareBalance'].sum().to_frame('ShareBalance').reset_index()
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost".format(user="root", pw="123456"))
stock.to_sql("ncf_data", con=engine, if_exists="replace", chunksize=1000, index=False, schema="StockRecommend")
print("*"*50 + "STOCK DATA FOR MODELING TO SQL = DONE" + "*"*50)
# For each acc has > min_stock_hold, extract 1 stock for test sample
df = stock
list_stock = df.ShareCode.unique().tolist()
df1 = df.groupby('main_account')['ShareCode'].count().to_frame('stock_count').reset_index().sort_values(by='stock_count', ascending=False)
df = pd.merge(df, df1, how='left', left_on='main_account', right_on='main_account')
df['rank'] = df.groupby('main_account')['ShareCode'].apply(lambda x: x.rank()) #add rank order based on main_account and sharecode
df['split_key'] = np.where(((df['stock_count'] == df['rank']) & (df['stock_count'] > min_stock_hold)), "Y", "N")
# Get train data
train = df[df['split_key'] != "Y"]
train.to_pickle('data/train.pickle')
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost".format(user="root", pw="123456"))
train.to_sql("ncf_train", con=engine, if_exists="replace", chunksize=1000, index=False, schema="StockRecommend")
print("*"*50 + "STOCK DATA FOR TRAINING MODEL TO SQL = DONE" + "*"*50)
# Get test data
test = df[df['split_key'] == "Y"]
test = test[~test['ShareCode'].isin(["YTC", "VCT", "VWS", "VTH", "VEC10801", "VMD", "VE3"])] # remove rows with stock in test but not in train
df2 = df.groupby('main_account')['ShareCode'].apply(list).to_frame('stock_hold').reset_index()  # list of stock hold
def get_random_stock_not_hold(list_stock, stock_hold):
    stock_not_hold = list(set(list_stock).symmetric_difference(set(stock_hold)))
    random_stock_not_hold = random.choices(stock_not_hold, k=random_test_size)
    return random_stock_not_hold
df2['stock_not_hold'] = df2['stock_hold'].apply(lambda x: get_random_stock_not_hold(list_stock, x)) # list of 'random_test_size' stock not hold
test = pd.merge(test, df2, how='left', left_on='main_account', right_on='main_account')
def get_test_sample(row):
    test_sample = row['ShareCode'].split() + row['stock_not_hold']
    return test_sample
test['test_sample'] = test.apply(get_test_sample, axis=1)
test.to_pickle('data/test.pickle')
test['stock_hold'] = test['stock_hold'].astype(str)   #convert list to str in order to import sql
test['stock_not_hold'] = test['stock_not_hold'].astype(str)   #convert list to str in order to import sql
test['test_sample'] = test['test_sample'].astype(str)   #convert list to str in order to import sql
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost".format(user="root", pw="123456"))
test.to_sql("ncf_test", con=engine, if_exists="replace", chunksize=1000, index=False, schema="StockRecommend")
print("*"*50 + "STOCK DATA FOR TESTING MODEL TO SQL = DONE" + "*"*50)










