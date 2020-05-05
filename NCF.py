import pandas as pd
import numpy as np
import pickle
import pandas_profiling
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from keras.models import Model, load_model
from keras import initializers
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.layers import Embedding, Input, merge, Flatten, concatenate, multiply, Dense
from keras.utils import plot_model, CustomObjectScope
from keras.callbacks import ModelCheckpoint
from utils import *
from evaluate import evaluate_model

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#
df = get_df_from_mysql(database_name="StockRecommend", table_name="ncf_train")
# df = pd.read_pickle('movie_len/train.pickle')
#
list_share = df.ShareCode.unique()
list_user = df.main_account.unique()
#
account_encoder = LabelEncoder()
df["account_encoder"] = account_encoder.fit_transform(df["main_account"])
account_dict = dict(zip(account_encoder.classes_, range(len(account_encoder.classes_))))
save_dict_to_pickle('data/account_dict.pickle', account_dict)
#
shareCode_encoder = LabelEncoder()
df["share_encoder"] = shareCode_encoder.fit_transform(df["ShareCode"])
share_dict = dict(zip(shareCode_encoder.classes_, range(len(shareCode_encoder.classes_))))
save_dict_to_pickle('data/share_dict.pickle', share_dict)
#
interaction_matrix = sp.dok_matrix((len(list_user) + 1, len(list_share) + 1), dtype=np.float32)
account_share_zip = list(zip(df["account_encoder"], df["share_encoder"]))
for i in range(len(account_share_zip)):
    account = account_share_zip[i][0]
    share = account_share_zip[i][1]
    interaction_matrix[account, share] = 1
save_dict_to_pickle('data/interaction_matrix.pickle', interaction_matrix)
# Hyper parameters
num_negatives = 1
num_epochs = 101
batch_size = 512
mf_dim = 8
layers = [8, 8, 8, 8]
reg_mf = 0
reg_layers = [0, 0, 0, 0]
learning_rate = 0.001
learner = Adam
verbose = 0
topK = 10
num_layer = len(layers)
evaluation_threads = 1
#
num_users, num_items = interaction_matrix.shape
#
def get_train_instances(interaction_matrix, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users = interaction_matrix.shape[0]
    for (u, i) in interaction_matrix.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in interaction_matrix:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


# Get model
user_input = Input(shape=(1,), dtype="int64", name="user_input")
item_input = Input(shape=(1,), dtype="int64", name="item_input")
MF_Embedding_User = Embedding(
    input_dim=num_users,
    output_dim=mf_dim,
    name="mf_embedding_user",
    embeddings_initializer="RandomNormal",
    embeddings_regularizer=l2(reg_mf),
    input_length=1,
)
MF_Embedding_Item = Embedding(
    input_dim=num_items,
    output_dim=mf_dim,
    name="mf_embedding_item",
    embeddings_initializer="RandomNormal",
    embeddings_regularizer=l2(reg_mf),
    input_length=1,
)
MLP_Embedding_User = Embedding(
    input_dim=num_users,
    output_dim=int(layers[0] / 2),
    name="mlp_embedding_user",
    embeddings_initializer="RandomNormal",
    embeddings_regularizer=l2(reg_layers[0]),
    input_length=1,
)
MLP_Embedding_Item = Embedding(
    input_dim=num_items,
    output_dim=int(layers[0] / 2),
    name="mlp_embedding_item",
    embeddings_initializer="RandomNormal",
    embeddings_regularizer=l2(reg_layers[0]),
    input_length=1,
)
mf_user_latent = Flatten()(MF_Embedding_User(user_input))
mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
#
mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])
mf_vector = multiply([mf_user_latent, mf_item_latent])  # element-wise multiply
for idx in range(1, num_layer):
    layer = Dense(
        layers[idx],
        activity_regularizer=l2(reg_layers[idx]),
        activation="relu",
        name="layer%d" % idx,
    )
    mlp_vector = layer(mlp_vector)
predict_vector = concatenate([mf_vector, mlp_vector])
prediction = Dense(
    1, activation="sigmoid", kernel_initializer="lecun_uniform", name="prediction")(predict_vector)
model = Model(inputs=[user_input, item_input], outputs=prediction)
model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
plot_model(model, to_file="model_architecture.png")
model.summary()
#
best_hr = 0
# checkpoint = ModelCheckpoint(
#     filepath="./model/model-{loss:.4f}.hdf5",
#     verbose=0,
#     save_best_only=True,
#     # save_weights_only= True,
#     monitor="loss",
#     mode="min",
# )
for epoch in range(num_epochs):
    # Fit model
    user_input_data, item_input_data, labels_data = get_train_instances(
        interaction_matrix, num_negatives
    )
    hist = model.fit(
        [np.array(user_input_data), np.array(item_input_data)],  # input
        np.array(labels_data),  # labels
        batch_size=batch_size,
        epochs=1,
        # callbacks=[checkpoint],
        shuffle=True,
    )
    if epoch % 10 == 0:
        hit_ratio_topK = evaluate_model(model, topK)
        print("Iteration %d : HR = %.4f, loss = %.4f" % (epoch, hit_ratio_topK, hist.history["loss"][0]))
        if hit_ratio_topK > best_hr:
            best_hr = hit_ratio_topK
        model_out_file = "model/NCF_%d_%s_%.4f_%d_%f.pkl" % (mf_dim, layers, hit_ratio_topK, num_negatives, hist.history["loss"][0])
        pickle.dump(model, open(model_out_file, 'wb'))
        # model.save(model_out_file)

