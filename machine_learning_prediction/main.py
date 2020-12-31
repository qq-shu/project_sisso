from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
from sklearn.pipeline import make_pipeline
import time
from scipy.stats import pearsonr,spearmanr
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib.pyplot import clabel
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import pandas as pd

# shuffled the dataset
data_array = np.loadtxt('shuffled_fatigue21.csv',encoding='utf-8-sig',delimiter=',')
# data_array = pd.read_csv('fatigue21.csv')
# data_array = data_array.sample(frac=1).reset_index(drop=True)
# data_array.to_csv('shuffer_fatigue21.csv', index=False)
# data_array = data_array.values

## build the priority model of GBDT,SVR,DT with GridSearch for selecting the best parameter for each prediction model, use 10 cross validation
model_pz = GridSearchCV(make_pipeline(preprocessing.StandardScaler(),GradientBoostingRegressor()),
                        param_grid={'gradientboostingregressor__n_estimators':[100,200,300,400,500],'gradientboostingregressor__learning_rate':[0.001,0.01,0.1,1],'gradientboostingregressor__max_depth':[1,2,3,4,5]},cv=10,scoring='r2',return_train_score=True)
# model_pz = GridSearchCV(make_pipeline(StandardScaler(),SVR()), param_grid={'svr__gamma': [0.01,0.05,0.1, 0.5, 1],'svr__C': [ 100, 150, 200]}, cv=10,scoring='r2',refit=True)
# model_pz = GridSearchCV(make_pipeline(StandardScaler(),DecisionTreeRegressor(min_samples_split=2,min_samples_leaf=1)),param_grid={'max_depth':[4,7,10],'decisiontreeregressor__ccp_alpha':
#                       [0.01,0.1,1.0]},cv=10,scoring='r2',refit=True)
# load the dataset
X = data_array[:,0:-1]
Y = data_array[:,-1]
# train the model
model_pz.fit(X,Y)
# predict Y under priority model as Y_pre
Y_pre=model_pz.predict(X)
# show the parameters of priority model
print(model_pz.best_params_)
# print out the evaluation indexes of testing for each model
print(model_pz.best_score_) # R² value of testing
print(model_pz.cv_results_['mean_test_score'])

# print out the evaluation indexes of training for each model
print(r2_score(Y,Y_pre))
# print(mean_squared_error(Y,Y_pre))
print(np.sqrt(mean_squared_error(Y,Y_pre)))
print(mean_absolute_error(Y,Y_pre))

# print the Ypre_Y figure of each model，which the Y axis represents the predicted fatigue strength and the X axis represents the real value
a=[0,1400]
fig = plt.gcf()
fig.set_size_inches(3.5, 2.5)
plt.xlabel('Real Value', fontdict={'family': 'Times New Roman', 'size': 10})
plt.ylabel('Predicted Value', fontdict={'family': 'Times New Roman', 'size': 10})
plt.xticks(fontproperties = 'Times New Roman', size = 10)
plt.yticks(fontproperties = 'Times New Roman', size = 10)
x_major_locator=MultipleLocator(200)
y_major_locator=MultipleLocator(200)
plt.xlim(0,1500)
plt.ylim(0,1500)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.scatter(Y, Y_pre, c='black',s = 8, alpha=0.6)
plt.plot(a,a,c='black')
# plt.text(20,1400,r'$feture\ number\ =\ 21$',fontdict={'size':'10','color':'black','family': 'Times New Roman'})
plt.show()
plt.savefig('Ypre_Y_021.png')
plt.cla()
