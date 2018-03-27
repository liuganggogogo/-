#encoding:utf-8
#@Time : 2018/3/23 9:01
#@Author : JackNiu
# 数据工具
import  numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from scipy import sparse

trainfile='E:\env\\tensorflow\Scripts\Kaggle\\almm\\round1_ijcai_18_train_20180301.txt'
demofile='demo.txt'
testdemo="E:\env\\tensorflow\Scripts\Kaggle\\almm\\testdemo.txt"
testfile ='E:\env\\tensorflow\Scripts\Kaggle\\almm\\round1_ijcai_18_test_a_20180301.txt'
featurelist=["item_id","item_brand_id","item_city_id","item_price_level","item_sales_level","item_collected_level","item_pv_level", "user_gender_id", "user_age_level", "user_occupation_id", "user_star_level","context_page_id","shop_review_num_level","shop_review_positive_rate","shop_star_level","shop_score_service","shop_score_delivery","shop_score_description"]



def loadClasAndRegData(filename,Mode="TRAIN"):
    data = pd.read_table(filename, header=0,delim_whitespace=True)
    if Mode =="TRAIN":
        X_data = data[featurelist]
        # X_data.astype('category')
        Y= data[['is_trade']]
        # for column in X_data.columns.values.tolist():
        #     X_data[column] = X_data[column].astype("category")
        X=np.array(X_data)
        Y=np.array(Y)
    if Mode=="Test":
        X_data = data[featurelist]
        X=np.array(X_data)
        Y=np.array(data[['instance_id']])


    return X,Y


    # train = data[featurelist]





def one_hot(test):
    labels =list(set(test))
    size = len(test)
    print(labels,size)
    tmparr = np.zeros((size,len(labels)))
    for index in range(size):
        tmparr[index,labels.index(test[index])]=1
    print(tmparr)
    return tmparr


def loadNNData(filename,Mode="Train"):
    data = pd.read_table(filename, header=0, delim_whitespace=True)
    # 在做一次格式控制 数字型的数据不需要变换
    if Mode == "Train":
        # data = data[featurelist]
        # item信息，忽略了 category 和 property属性
        data_onehot = one_hot(data['item_id'])
        data['category']= data['item_category_list'].apply(lambda x:x.split(';')[1])
        data_onehot = np.concatenate((data_onehot, one_hot(data['category'])), axis=1)
        data_onehot = np.concatenate((data_onehot, one_hot(data['item_brand_id'])), axis=1)
        data_onehot = np.concatenate((data_onehot, one_hot(data['item_city_id'])), axis=1)
        data_onehot = np.concatenate((data_onehot, np.reshape(np.array(data['item_price_level']),(-1,1))), axis=1)
        data_onehot = np.concatenate((data_onehot, np.reshape(np.array(data['item_sales_level']),(-1,1))), axis=1)
        data_onehot = np.concatenate((data_onehot, np.reshape(np.array(data['item_collected_level']),(-1,1))), axis=1)
        data_onehot = np.concatenate((data_onehot, np.reshape(np.array(data['item_pv_level']),(-1,1))), axis=1)

        # user 信息，user_id， user_gender_id,user_age_level,user_occupation_id,user_star_level
        data_onehot = np.concatenate((data_onehot, np.reshape(np.array(data['user_gender_id']), (-1, 1))), axis=1)
        data_onehot = np.concatenate((data_onehot, np.reshape(np.array(data['user_age_level']), (-1, 1))), axis=1)
        data_onehot = np.concatenate((data_onehot, np.reshape(np.array(data['user_occupation_id']), (-1, 1))), axis=1)
        data_onehot = np.concatenate((data_onehot, np.reshape(np.array(data['user_star_level']), (-1, 1))), axis=1)

        # context 属性 忽略 context_id, context_timestamp,predict_category_property
        data_onehot = np.concatenate((data_onehot, one_hot(data['context_page_id'])), axis=1)

        # shop , shop_id, shop_review_num_level,shop_review_positive_rate,shop_star_level,shop_score_service,shop_score_delivery,shop_score_description
        data_onehot = np.concatenate((data_onehot, np.reshape(np.array(data['shop_review_num_level']), (-1, 1))), axis=1)
        data_onehot = np.concatenate((data_onehot, np.reshape(np.array(data['shop_review_positive_rate']), (-1, 1))), axis=1)
        data_onehot = np.concatenate((data_onehot, np.reshape(np.array(data['shop_star_level']), (-1, 1))), axis=1)
        data_onehot = np.concatenate((data_onehot, np.reshape(np.array(data['shop_score_service']), (-1, 1))), axis=1)
        data_onehot = np.concatenate((data_onehot, np.reshape(np.array(data['shop_score_delivery']), (-1, 1))), axis=1)
        data_onehot = np.concatenate((data_onehot, np.reshape(np.array(data['shop_score_description']), (-1, 1))), axis=1)

        # is_trade
        data_onehot = np.concatenate((data_onehot, np.reshape(np.array(data['is_trade']), (-1, 1))),axis=1)

    return data_onehot


def time2cov(time_):
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_))


def loadSklearnData(filename):
    data = pd.read_table(filename, header=0, delim_whitespace=True)
    print('预处理')
    print('item_category_list_ing')
    for i in range(3):
        data['category_%d' % (i)] = data['item_category_list'].apply(

            lambda x: x.split(";")[i] if len(x.split(";")) > i else " "
        )
    del data['item_category_list']
    print('item_property_list_ing')
    for i in range(30):
        data['property_%d' % (i)] = data['item_property_list'].apply(

            lambda x: x.split(";")[i] if len(x.split(";")) > i else " "
        )
    del data['item_property_list']
    print('context_timestamp_ing')
    data['context_timestamp'] = data['context_timestamp'].apply(time2cov)
    # del data['context_timestamp']
    print('predict_category_property_ing_0')

    for i in range(3):
        data['predict_category_%d' % (i)] = data['predict_category_property'].apply(

            lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else " "

        )

    return data

def loadSklearnData_0():
    train = loadSklearnData(trainfile)
    print(train.shape)
    test = loadSklearnData(testfile)
    print(test.shape)
    val = train[train['context_timestamp']>'2018-09-23 23:59:59']
    tra =train[train['context_timestamp'] <= '2018-09-23 23:59:59']

    del tra['context_timestamp']
    del val['context_timestamp']
    del test['context_timestamp']

    train_index = tra.pop('instance_id')
    train_label = tra.pop('is_trade')
    val_index = val.pop('instance_id')
    val_label = val.pop('is_trade')

    test_index = test.pop('instance_id')
    print(test.columns)
    enc = OneHotEncoder()
    lb = LabelEncoder()
    feat_set = list(test.columns)
    del train

    for i, feat in enumerate(feat_set):
        tmp = lb.fit_transform((list(tra[feat])+list(val[feat])+list(test[feat])))
        enc.fit(tmp.reshape(-1, 1))
        x_train = enc.transform(lb.transform(tra[feat]).reshape(-1, 1))
        x_val = enc.transform(lb.transform(val[feat]).reshape(-1,1))
        x_test = enc.transform(lb.transform(test[feat]).reshape(-1, 1))
        if i == 0:
            X_train, X_test,X_val = x_train, x_test,x_val

        else:
            X_train, X_test,X_val = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test)),sparse.hstack((X_val,x_val))

    return X_train,train_label,X_val,val_label,X_test,test_index











