#encoding:utf-8
#@Time : 2018/3/26 9:18
#@Author : JackNiu

import lightgbm as lgb
import  ml_almm.misc_utils as misc_utils
import pandas as pd


gbm = lgb.LGBMRegressor(objective='binary',

                        num_leaves=64,

                        learning_rate=0.01,

                        n_estimators=3000)

X_train,train_label,X_val,val_label,X_test,test_index=misc_utils.loadSklearnData_0()
gbm.fit(X_train, train_label,

        eval_set=[(X_val, val_label)],

        eval_metric='binary_logloss',

        early_stopping_rounds=50)

print('Start predicting...')
y_sub_1 = gbm.predict(X_test)

sub = pd.DataFrame()

sub['instance_id'] = list(test_index)

sub['predicted_score'] = list(y_sub_1)

sub.to_csv('./result/2018031802.txt',sep=" ",index=False)