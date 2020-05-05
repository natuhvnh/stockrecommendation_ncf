import pandas as pd
import mysql.connector
import requests
from sqlalchemy import create_engine
import pickle


def get_bearer_token():
    account = {"Username": "tu.na@financialdeepmind.com", "Password": "Du1Lieu2That3"}
    response = requests.post(
        "https://10.32.59.16/api/account/authenticate", json=account, verify=False
    )
    bearer_token = response.text
    return bearer_token


def get_data_from_api(data_name, year, month, day, bearer_token):
    base_url = "https://10.32.59.16/api/"
    if data_name == "customers":
        url = base_url + data_name
    else:
        url = base_url + data_name + "?date=" + str(year) + "-" + str(month) + "-" + str(day)
    response = requests.get(
        url, headers={"Authorization": "Bearer " + bearer_token}, verify=False,
    )
    return response.json()


def api_data_to_mysql(year, month, day):
    table_name = "T" + str(month) + "_" + str(year)
    bearer_token = get_bearer_token()
    engine = create_engine("mysql+pymysql://{user}:{pw}@localhost".format(user="root", pw="123456"))
    #
    cash = get_data_from_api("cashes", year=year, month=month, day=day, bearer_token=bearer_token)
    cash = pd.DataFrame.from_dict(cash)
    cash.to_sql(table_name, con=engine, if_exists="replace", chunksize=1000, index=False, schema="Cash")
    print("*"*50 + "CASH DATA FROM API TO MYSQL = DONE" + "*"*50)
    #
    customers = get_data_from_api("customers", year=year, month=month, day=day, bearer_token=bearer_token)
    customers = pd.DataFrame.from_dict(customers)
    customers['Accounts'] = customers['Accounts'].astype(str)   #convert list to str in order to import sql
    customers.to_sql(table_name, con=engine, if_exists="replace", chunksize=1000, index=False, schema="Customer")
    print("*"*50 + "CUSTOMERS DATA FROM API TO MYSQL = DONE" + "*"*50)
    #
    derivative = get_data_from_api("derivativeaccounts", year=year, month=month, day=day, bearer_token=bearer_token)
    derivative = pd.DataFrame.from_dict(derivative)
    derivative['Positions'] = derivative['Positions'].astype(str)   #convert list to str in order to import sql
    derivative.to_sql(table_name, con=engine, if_exists="replace", chunksize=1000, index=False, schema="Derivative")
    print("*"*50 + "DERIVATIVE DATA FROM API TO MYSQL = DONE" + "*"*50)
    #
    loan = get_data_from_api("loans", year=year, month=month, day=day, bearer_token=bearer_token)
    loan = pd.DataFrame.from_dict(loan)
    loan.to_sql(table_name, con=engine, if_exists="replace", chunksize=1000, index=False, schema="Loan")
    print("*"*50 + "LOAN DATA FROM API TO MYSQL = DONE" + "*"*50)
    #
    stock = get_data_from_api("stocks", year=year, month=month, day=day, bearer_token=bearer_token)
    stock = pd.DataFrame.from_dict(stock)
    stock.to_sql(table_name, con=engine, if_exists="replace", chunksize=1000, index=False, schema="Stock")
    print("*"*50 + "STOCK DATA FROM API TO MYSQL = DONE" + "*"*50)
    #
    price = get_data_from_api("marketPrices", year=year, month=month, day=day, bearer_token=bearer_token)
    price = pd.DataFrame.from_dict(price)
    price.to_sql(table_name, con=engine, if_exists="replace", chunksize=1000, index=False, schema="Market_price")
    print("*"*50 + "PRICE DATA FROM API TO MYSQL = DONE" + "*"*50)
    return print("DONE!")




def get_df_from_mysql(database_name, table_name):
    mydb = mysql.connector.connect(host="localhost", user="root", passwd="123456")
    query = "select * from {}.{}".format(database_name, table_name)
    df = pd.read_sql(query, con=mydb)
    return df


def split_type_and_main_account(dataframe, account_column):
    dataframe["main_account"] = dataframe[account_column].apply(lambda x: x[:-1])
    dataframe["type_account"] = dataframe[account_column].apply(lambda x: x[-1:])
    return dataframe

def save_dict_to_pickle(file_path, dict_data):
    pickle_out = open(file_path,"wb")
    pickle.dump(dict_data, pickle_out)
    pickle_out.close()

def load_dict_from_pickle(file_path):
    pickle_in = open(file_path,"rb")
    dict_data = pickle.load(pickle_in)
    return dict_data