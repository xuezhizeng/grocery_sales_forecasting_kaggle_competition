# -*- coding: utf-8 -*-
"""
@Createon : 2022-05-19 version 1.0
@Author : Xuezhi Zeng
@Purpose: this code is to build a model to forecast grocery sales. It builds several different models to forecast grocery sales between 2017-08-16 and 2017-08-31 for copoerate favarite retail.
the forecasting outcome is saved in forecast_outcome.csv
all models' evaluation outcome is saved in model_evaluation_outcome.csv
five evaluaton metrics are used here (i.e., RMSE,MAE, R2, NWRMSE,NWRMSLE)

"""
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from time import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import xgboost
from sklearn.linear_model import LinearRegression, SGDRegressor

from tensorflow.keras.layers import Input, Embedding, dot, Dot, add, Flatten, concatenate, Dropout, Dense,BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorboard.plugins import projector

from sklearn.manifold import TSNE
from eli5 import show_weights


def process_dataframe(df):
    '''
    process the date column in a dataframe to extract year, month, dayofweek etc., as features in the model
    Args:
        df:a dataframe
    Returns:
    '''
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofmonth'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['onpromotion'] = df['onpromotion'].map({'False': 0, 'True': 1})
    df['perishable'] = df['perishable'].map({0: 1.0, 1: 1.25})
    df = df.fillna(-1)
    return df


def label_transform(data, tables, columns):
    '''
    Args:
        data: a dictionary containing train data, test sales data etc.,
        tables: a list containing specific dataset
        columns: a list containing fields in a dataset

    Returns: a dictionary(key: item_nbr, value: index)
    '''
    le = preprocessing.LabelEncoder()
    item_encode = None
    if len(tables) != 1:
        le.fit(data[tables[0]][columns[0]])
        #item_encode is used for visualizing items' embedding features later
        item_encode = {each: index for index, each in enumerate(le.classes_)}
        for table in tables:
            data[table][columns[0]] = le.transform(data[table][columns[0]])
    else:
        for column in columns:
            data[tables[0]][column] = le.fit_transform(data[tables[0]][column])
    return item_encode

def fillOilMissingValue(data):
    '''
    fill with all the missing dates firstly and then fill all missing prices
    with moving avarage in order to merge the oil pricing data
    Args: data dictionary

    return: data dictionary
    '''
    dates = data['train'].date
    dates = dates.append(data['test'].date)
    unique_dates = pd.DataFrame({'date': dates.unique()})
    oil = data["oil"]
    oil = oil.merge(unique_dates, on='date', how='outer')
    oil = oil.sort_values('date')
    oil = oil.fillna(oil['dcoilwtico'].rolling(10, min_periods=1, center=True, win_type='gaussian').mean(std=1).to_frame())
    data["oil"] = oil
    return data

def cal_RMSE(y_true, y_pred):
    #for calculating root mean square error
    return metrics.mean_squared_error(y_true, y_pred, squared=False)


def cal_MAE(y_true, y_pred):
    # for calculating mean absolute error
    return metrics.mean_absolute_error(y_true, y_pred)


def cal_R2(y_true, y_pred):
    return metrics.r2_score(y_true, y_pred )


def cal_NWRMSE(y_true, y_pred, weight):
    #for calculating normalized weighted root mean square error
    weight = np.array(weight)
    return metrics.mean_squared_error(y_true, y_pred, sample_weight=weight, squared=False)

def log_transform(arr):
    ''' Transforms array taking log(x+1) for every element (x) '''

    new_arr=[]
    for value in arr:
        if float(value) > 0:
            value=np.log1p(float(value))
        else:
            value=0
        new_arr.append(value)
    return np.array(new_arr)


def cal_NWRMSLE(y_true,y_pred,weight):
    '''
    Calculates Normalized Weighted Root Mean Squared Logarithmic Error (nwrmsle)
    we use log_transform to process y_true and y_pred to aovid invalid log calculation
    when negative values appear
    returns nwrmsle
    '''
    temp=(log_transform(y_true) - log_transform(y_pred))**2
    temp = temp * weight
    nwrmsle = np.sqrt(temp.sum() / weight.sum() )
    return nwrmsle


class label_encoding_data(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("label encoding data -> start")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        item_encode = label_transform(X, ["item", "train", "test"], ["item_nbr"])
        _ = label_transform(X, ["store", "transaction", "train", "test"], ["store_nbr"])
        _ = label_transform(X, ["item"], ["family", "class"])
        _ = label_transform(X, ["store"], ["city", "state", "type", "cluster"])
        _ = label_transform(X, ["holiday"], ['type', 'locale', 'transferred'])
        return [X, item_encode]


class merge_data(BaseEstimator, TransformerMixin):
    '''
    a class to merge train, item, transaction, store, holiday and oil dataset
    '''

    def __init__(self):
        print("merge_data -> start")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        train_item = X[0]["train"].merge(X[0]["item"], how='left', on=['item_nbr'])
        train_item_transaction = train_item.merge(X[0]["transaction"], how='left', on=['date', 'store_nbr'])
        train_item_transaction_store = train_item_transaction.merge(X[0]["store"], how='left', on=['store_nbr'])
        train_item_transaction_store_holiday = train_item_transaction_store.merge(X[0]["holiday"], how='left',
                                                                                  on=['date'])
        train_item_transaction_store_holiday_oil = train_item_transaction_store_holiday.merge(X[0]["oil"], how='left',
                                                                                              on=['date'])

        data_df = train_item_transaction_store_holiday_oil.copy(deep=True)
        data_df.rename(columns={'type_x': 'store_type', 'type_y': 'holiday_type'}, inplace=True)
        data_df.drop(['id'], axis=1, inplace=True)

        test_item = X[0]["test"].merge(X[0]["item"], how='left', on=['item_nbr'])
        test_item_transaction = test_item.merge(X[0]["transaction"], how='left', on=['date', 'store_nbr'])
        test_item_transaction_store = test_item_transaction.merge(X[0]["store"], how='left', on=['store_nbr'])
        test_item_transaction_store_holiday = test_item_transaction_store.merge(X[0]["holiday"], how='left',
                                                                                on=['date'])
        test_item_transaction_store_holiday_oil = test_item_transaction_store_holiday.merge(X[0]["oil"], how='left',
                                                                                            on=['date'])
        test_df = test_item_transaction_store_holiday_oil.copy(deep=True)
        test_df.rename(columns={'type_x': 'store_type', 'type_y': 'holiday_type'}, inplace=True)
        test_df.drop(['id'], axis=1, inplace=True)
        return [data_df, test_df, X[1]]


class transform_data(BaseEstimator, TransformerMixin):
    '''
    a class to transform unit_sales and transactions values of train and test dataset using log1p
    '''

    def __init__(self):
        print("process_data -> start")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[0] = process_dataframe(X[0])
        X[1] = process_dataframe(X[1])
        X[0].drop(['date'], axis=1, inplace=True)
        X[1].drop(['date'], axis=1, inplace=True)
        return X[0], X[1], X[2]


def count_items_stores(items, stores):
    """
    Args：
        items: item dataset
        stores: stores dataset
    Returns：
        the legnth of unique vaues of the relavent features in each dataset
    """
    n_items = len(items.item_nbr.unique())
    n_class = len(items['class'].unique())
    n_family = len(items['family'].unique())
    n_store = len(stores['store_nbr'].unique())
    n_type = len(stores['type'].unique())
    n_city = len(stores['city'].unique())
    n_state = len(stores['state'].unique())
    n_cluster = len(stores['cluster'].unique())
    return n_class, n_store, n_type, n_city, n_state, n_cluster, n_items, n_family


def convertDF2Tensor(df):
    """
    Args:
        df(DataFrame):  data
        Output:
    Returns:
        x: features
        y: target
    """
    # At this stage, three embedding features(i.e., 'store_nbr', 'item_nbr', 'family') are used here.
    # More embedding features can be introduced when need
    features = ['store_nbr', 'item_nbr', 'family']

    target_class = 'unit_sales'

    x = []
    y = []
    for f in features:
        x.append(df[f])
    if target_class is not None:
        y = df[target_class]
    return x, y


def embedding_predict(embeddingmodel, x):
    '''
    Args:
        embeddingmodel: an embedding model generated by a neural network
        x: the object that will be predicted

    '''
    y = embeddingmodel.predict(x, batch_size=128, verbose=1, workers=4, use_multiprocessing=True)
    y = y.reshape(-1)
    y = np.clip(y, 0, max(0, y.max()))
    return y


def evaluateModel(model_dict, x_train, y_train, x_test, y_test, model_evaluation_outcome, modeltype, perishable_weight_train,
                  perishable_weight_test):
    '''
    Args:
        model_dict: a dictionary containing model's name and model itself
        x_train: labels of training dataset
        y_train: targets of training dataset
        x_test: labels of training dataset
        y_test: targets of training dataset
        model_evaluation_outcome: store the performance for all models and save to .xslx file
        modeltype: it consists of three types of model (i.e., normal,embedding, use_embedding_features)
        perishable_weight_train: the weight matrix in training set
        perishable_weight_test: the weight matrix in testing set

    '''
    t0 = time()
    model_name = model_dict["name"]
    model = model_dict["model"]

    if (modeltype == "normal"):
        if type(x_train) != np.ndarray:
            model.fit(x_train.values, y_train.values)
            y_train_pred = model.predict(x_train.values)
            y_test_pred = model.predict(x_test.values)
        else:
            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
    elif modeltype == "Embedding":
        x_train_tensor, y_train_tensor = convertDF2Tensor(x_train.join(y_train))
        x_test_tensor, y_test_tensor = convertDF2Tensor(x_test.join(y_test))
        embedding_model, history = train_embedding_model(x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)
        y_train_pred = embedding_predict(embedding_model, x_train_tensor)
        y_test_pred = embedding_predict(embedding_model, x_test_tensor)

    else:

        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

    t1 = time() - t0
    rmse_train = cal_RMSE(y_train.values, y_train_pred)
    mae_train = cal_MAE(y_train.values, y_train_pred)
    r2_train = cal_R2(y_train.values, y_train_pred)
    nwrmse_train = cal_NWRMSE(y_train.values, y_train_pred, perishable_weight_train.values)
    nwrmsle_train = cal_NWRMSLE(y_train.values, y_train_pred, perishable_weight_train.values)


    rmse_test = cal_RMSE(y_test.values, y_test_pred)
    mae_test = cal_MAE(y_test.values, y_test_pred)
    r2_test = cal_R2(y_test.values, y_test_pred)
    nwrmse_test = cal_NWRMSE(y_test.values, y_test_pred, perishable_weight_test.values)
    nwrmsle_test = cal_NWRMSLE(y_test.values, y_test_pred, perishable_weight_test.values)


    model_evaluation_outcome = model_evaluation_outcome.append(
        {'Model Name': model_name, "RMSE_01_train": rmse_train, "MAE_02_train": mae_train,
         "WRMSE_03_train": nwrmse_train, "WRMSLE_04_train": nwrmsle_train,"R2_train":r2_train,
         "RMSE_01_test": rmse_test, "MAE_02_test": mae_test, "WRMSE_03_test":nwrmse_test ,
         "WRMSLE_04_test": nwrmsle_test, "R2_test":r2_test}, ignore_index=True)

    if (model_name != "Embedding"):
        return model, model_evaluation_outcome
    else:
        return embedding_model, model_evaluation_outcome


msle = tf.keras.losses.MeanSquaredLogarithmicError()


def generate_embedding_model():
    store_input = Input(shape=(1,), dtype='int64', name='store_embedding')
    store = Embedding(n_store, 20, input_length=1, embeddings_regularizer=l2(1e-4), )(store_input)

    item_input = Input(shape=(1,), dtype='int64', name='item_embedding')
    item = Embedding(n_items, 100, input_length=1, embeddings_regularizer=l2(1e-4))(item_input)

    family_input = Input(shape=(1,), dtype='int64', name='cluster_embedding')
    family = Embedding(n_family, 15, input_length=1, embeddings_regularizer=l2(1e-4))(family_input)

    x = concatenate([store, item, family])
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    nn = Model([store_input, item_input, family_input], x)
    nn.compile(Adam(0.001), loss='mse', metrics=['mae', msle])
    nn.summary()
    return nn


def train_embedding_model(x_train, y_train, x_val, y_val):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    embedding_model = generate_embedding_model()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = embedding_model.fit(x_train, y_train, batch_size=512, epochs=10, validation_data=(x_val, y_val),
                                  callbacks=[callback, tensorboard_callback])

    return embedding_model, history


def display_embedding_tsne(encoder, embeddings, name):
    vectors = [embeddings[word] for key, word in encoder.items()]

    word_labels = [word for word in encoder]
    word_vec_zip = zip(word_labels, vectors)

    word_vec_dict = dict(word_vec_zip)
    df = pd.DataFrame.from_dict(word_vec_dict, orient='index')

    tsne = TSNE(perplexity=65, n_components=2, random_state=42)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(df)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.figure(figsize=(16, 8))
    plt.plot(x_coords, y_coords, 'ro')

    for label, x, y in zip(df.index, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(1, 1), textcoords='offset points', color='g')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()


def forecast_and_save_to_csv(test_df, embedding_model):
    features = ['store_nbr', 'item_nbr', 'family']
    X = []
    for f in features:
        X.append(test_df[f])

    y_test_tag = embedding_model.predict(X, verbose=1)
    y_test_tag = np.clip(y_test_tag, 0, max(0, y_test_tag.max()))
    predictions = pd.DataFrame({'id': test_df['id'].values, 'unit_sales': y_test_tag.reshape(1, -1)[0]})
    predictions.to_csv('forecast_outcome.csv', index=False)


if __name__ == "__main__":

    log_dir = "./embedding_log"
    dtypes = {'id': 'uint32', 'onpromotion': str}
    data = {
        'train': pd.read_csv('data/train.csv', dtype=dtypes, parse_dates=['date']),
        'test': pd.read_csv('data/test.csv', dtype=dtypes, parse_dates=['date']),
        'item': pd.read_csv('data/items.csv'),
        'store': pd.read_csv('data/stores.csv'),
        'transaction': pd.read_csv('data/transactions.csv', parse_dates=['date']),
        'holiday': pd.read_csv('data/holidays_events.csv', dtype={'transferred': str}, parse_dates=['date']),
        'oil': pd.read_csv('data/oil.csv', parse_dates=['date']),
    }

    data['train']['item_nbr'] = data['train']['item_nbr'].astype(str)
    data['train']['store_nbr'] = data['train']['store_nbr'].astype(str)
    data['test']['item_nbr'] = data['test']['item_nbr'].astype(str)
    data['test']['store_nbr'] = data['test']['store_nbr'].astype(str)
    data['item']['item_nbr'] = data['item']['item_nbr'].astype(str)
    data['item']['class'] = data['item']['class'].astype(str)
    data['item']['family'] = data['item']['family'].astype(str)
    data['store']['store_nbr'] = data['store']['store_nbr'].astype(str)
    data['store']['cluster'] = data['store']['cluster'].astype(str)
    data['transaction']['store_nbr'] = data['transaction']['store_nbr'].astype(str)

    # select sales data after August 15th every year between 2013 and 2017 for the purpose of matching the date in test dataset (forecasting sales from 2017-08-16).
    data['train'] = data['train'][(data['train']['date'].dt.month == 8) & (data['train']['date'].dt.day > 15)]
    data['holiday'] = data['holiday'][['date', 'type', 'locale', 'transferred']]

    n_class, n_store, n_type, n_city, n_state, n_cluster, n_items, n_family = count_items_stores(data['item'],data['store'])

    # use pipeline to make it clear and easy to change features representations
    pipe_processing = Pipeline([
        ('label_encoding_data', label_encoding_data()),
        ('merge_data', merge_data()),
        ('transform_data', transform_data()),
    ])

    #data_df will be spilt into training and testing dataset
    #test_df will be predicted
    data = fillOilMissingValue(data)

    data_df, test_df, item_encode = pipe_processing.fit_transform(data)

    features = ['unit_sales','store_nbr', 'item_nbr', 'onpromotion', 'family',
       'class', 'perishable', 'city', 'state', 'store_type',
       'cluster', 'dcoilwtico', 'year', 'month',
       'dayofmonth', 'dayofweek']

    #at this stage, the above 16 features will be fed into forecasting models.
    #we could easily add more features when needed
    data_df = data_df[features]
    test_df = test_df[features[1:]]

    # split it according to our feature engineering (unit_sales is dependent variable )
    X = data_df.drop(['unit_sales'], axis=1)
    Y = data_df[['unit_sales']]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    x_train = x_train.reset_index()
    x_train = x_train.drop(columns=["index"])
    y_train = y_train.reset_index()
    y_train = y_train.drop(columns=["index"])
    x_test = x_test.reset_index()
    x_test = x_test.drop(columns=["index"])
    y_test = y_test.reset_index()
    y_test = y_test.drop(columns=["index"])

    # this is for feeding into
    x_train_tensor, y_train_tensor = convertDF2Tensor(x_train.join(y_train))
    x_test_tensor, y_test_tensor = convertDF2Tensor(x_test.join(y_test))

    model_evaluation_outcome = pd.DataFrame(
        columns=['Model Name'])

    lr_model, model_evaluation_outcome = evaluateModel({"name": "Linear Regression", "model": LinearRegression()}, x_train, y_train,
                                    x_test, y_test, model_evaluation_outcome, 'normal', x_train['perishable'], x_test['perishable'])

    show_weights(lr_model, feature_names=list(x_train.columns))

    xgb_model, model_evaluation_outcome = evaluateModel({"name": "XGBRegression", "model": xgboost.XGBRegressor(random_state=42)}, x_train,
                                     y_train, x_test, y_test, model_evaluation_outcome, 'normal', x_train['perishable'],
                                     x_test['perishable'])

    show_weights(xgb_model, feature_names=list(x_train.columns))

    #this step takes very long time, be careful
    # rf_model_default, model_evaluation_outcome = evaluateModel({"name":"Random Forest Regression", "model":RandomForestRegressor(random_state=42)},x_train, y_train,x_test,y_test , stats,'normal',x_train['perishable'],x_test['perishable'])

    rf_model_adjusted, model_evaluation_outcome = evaluateModel({"name": "Random Forest Regression self parameters",
                                              "model": RandomForestRegressor(n_estimators=10, max_depth=5,
                                                                             min_samples_leaf=3, random_state=42)},
                                             x_train, y_train, x_test, y_test, model_evaluation_outcome, 'normal', x_train['perishable'],
                                             x_test['perishable'])

    show_weights(rf_model_adjusted, feature_names=list(x_train.columns))

    model_embedding, model_evaluation_outcome = evaluateModel({"name": "Embedding", "model": train_embedding_model}, x_train, y_train,
                                           x_test, y_test, model_evaluation_outcome, 'Embedding', x_train['perishable'],
                                           x_test['perishable'])

    # Save the embedding result in order to be displayed by tensorboard
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for itemID, seqno in item_encode.items():
            f.write("{}\n".format(itemID))

    # Save the weights we want to analyze as a variable
    weights_item_embedding = tf.Variable(model_embedding.get_weights()[1])
    weights_store_embedding = tf.Variable(model_embedding.get_weights()[0])
    # Create a checkpoint from embedding, the filename and key are the name of the tensor.
    checkpoint_item = tf.train.Checkpoint(embedding=weights_item_embedding)
    checkpoint_store = tf.train.Checkpoint(embedding=weights_store_embedding)
    checkpoint_item.save(os.path.join(log_dir, "item_embedding.ckpt"))
    checkpoint_store.save(os.path.join(log_dir, "store_embedding.ckpt"))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)

    store_encode = store_enc = {v: k for (k, v) in enumerate(data['store'].store_nbr.unique())}
    store_embeddings = model_embedding.get_weights()[0]
    item_embeddings = model_embedding.get_weights()[1]
    family_embeddings = model_embedding.get_weights()[2]
    display_embedding_tsne(store_encode, store_embeddings, 'store')
    display_embedding_tsne(item_encode, item_embeddings, 'item')

    le = preprocessing.LabelEncoder()
    family_encode = []
    df_item = pd.read_csv('data/items.csv')
    le.fit(df_item["family"])
    family_encode = {each: index for index, each in enumerate(le.classes_)}
    display_embedding_tsne(family_encode,family_embeddings,'family')

    x_train_tensor, y_train_tensor = convertDF2Tensor(x_train.join(y_train))
    x_test_tensor, y_test_tensor = convertDF2Tensor(x_test.join(y_test))

    embedding_extracted = Model(model_embedding.inputs, model_embedding.layers[-3].output)
    features = embedding_extracted.predict(x_train_tensor)
    features_test = embedding_extracted.predict(x_test_tensor)

    xgb_model_embedding_features, model_evaluation_outcome = evaluateModel(
        {"name": "XGBRegression_embedding_features", "model": xgboost.XGBRegressor(random_state=42)}, features, y_train,
        features_test, y_test, model_evaluation_outcome, 'normal', x_train['perishable'], x_test['perishable'])

    test_df['id'] = data['test']['id']
    forecast_and_save_to_csv(test_df, model_embedding)

    model_evaluation_outcome.to_csv("model_evaluation_outcome.csv")


