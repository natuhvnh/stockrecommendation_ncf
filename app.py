import pandas as pd 
import numpy as np
import pickle
from flask_restplus import Api, Resource, fields
from flask import Flask, request, jsonify
from utils import load_dict_from_pickle

flask_app = Flask(__name__)
app = Api(app = flask_app,
          version = "1.0",
          title = "Name Recorder",
          description = "Manage names of various users of the application")

name_space = app.namespace('main', description='Manage names')
body_require = app.model('main_account',
                  {'main_account': fields.String(required = True, description="Stock account", help="Cannot be blank.",
                  example="0023387d6b544f53a9306495d5576cae")})
#load data
model = pickle.load(open('model/NCF_8_[8, 8, 8, 8]_0.6663_1_0.282744.pkl', 'rb'))
interaction_matrix = load_dict_from_pickle('data/interaction_matrix.pickle')
account_dict = load_dict_from_pickle('data/account_dict.pickle')
share_dict = load_dict_from_pickle('data/share_dict.pickle')
#
topK = 10

@name_space.route("/")
class MainClass(Resource):
  @app.doc(responses={ 200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error' })
  def get(self):
    return {"status": "Person retrieved"}
  
  @app.doc(responses={ 200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error' })
  @app.expect(body_require)
  def post(self):
    data = request.json['main_account']
    main_account_encode = account_dict[data]
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

# Run terminal: FLASK_APP=app.py flask run
if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', debug=True, use_reloader=False)  # `use_reloader=False` = flask not reload twice, but not reload when changes
