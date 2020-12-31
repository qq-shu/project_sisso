import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sko.PSO import PSO
from sklearn.metrics import make_scorer
current_model = None

def read_data():
    # read csv file
    raw = pd.read_csv('psoall_data.csv').dropna().astype('float64').values
    features = raw[:, :-1]
    targets = raw[:, -1]
    return features, targets
    ## split the data to training and test set(for small sample，can train all the data to see pso search result)
    # x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.3)
    # return x_train, x_test, y_train, y_test

def create_and_fit_model(x_train, y_train):
    # define model（21-dimesional input，one-dimesional output）
    # Here use StandardScaler() to do normalization，and the pso search result will show the nomalized values of x afterwards
    # but with saving the normalized StandardScaler, we can recover the nomalized data to feature values as the result
    # model_pz = GridSearchCV(make_pipeline(StandardScaler(),SVR()), param_grid={'svr__gamma': [0.01,0.05,0.1, 0.5, 1],'svr__C': [ 100, 150, 200]}, cv=10,scoring='r2',refit=True)
    # model_pz = GridSearchCV(make_pipeline(StandardScaler(),DecisionTreeRegressor(min_samples_split=2,min_samples_leaf=1)),param_grid={'max_depth':[4,7,10],'decisiontreeregressor__ccp_alpha':
    #                       [0.01,0.1,1.0]},cv=10,scoring='r2',refit=True)
    model_pz = GridSearchCV(make_pipeline(preprocessing.StandardScaler(),GradientBoostingRegressor()),
                        param_grid={'gradientboostingregressor__n_estimators':[100,200,300,400,500],'gradientboostingregressor__learning_rate':[0.001,0.01,0.1,1],
                         'gradientboostingregressor__max_depth':[1,2,3,4,5]},cv=10,scoring='r2',refit=True)
    # train model
    model_pz.fit(x_train, y_train)
    print(model_pz.best_params_)
    global current_model
    current_model = model_pz

def calc_func(x):
    ## print the log
    # print('current x = {}'.format(x))
    # define the search range with the indexes of the features which should do search(the index values start from 0)
    # 0:THT, 1:THQCr, 2:NT, 3:Ct, 4:Dt, 5:QmT, 6:Tt, 7:TCr, 8:Si, 9:Mn, 10:P, 11:S, 12:Ni, 13:Mo, 14:CT, 15:THT,16:DT,17:TT,18:Cu,19:C,20:Cr
    search_range = [2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16, 17, 18,19,20]
    # As the first two columns of 21-dimesional features in the based dataset value 0, so here perform pso search on the remained 19 features
    assert len(x) == len(search_range), 'the length of input x and search range are differ, pls. check'
    # define constant dataset, here adopt the data of the maximal fatigue strength as dataset model
    data_template = [[0, 0,930, 540, 15, 140, 120, 0.5, 0.35, 0.51, 0.008, 0.012, 0.11, 0.18, 930, 30, 850, 160, 1.69, 0.23, 0.55]]
    # replace dataset
    for i in range(len(search_range)):
        data_template[0][search_range[i]] = x[i]
    # transform to nparray
    pso_x = np.array(data_template)
    # load model
    global current_model
    if current_model == None:
        print('pls. call create_and_fit_model() to train model first')
        return
    else:
        return -current_model.predict(pso_x)

def calc_pso():
    # 'num' means search range，the length of parameter 'search_range' in calc_func should keep same with it
    num = 19
    # As the first two columns of 21-dimesional features in the based dataset value 0, so here perform pso search on the remained 19 features 
    pso = PSO(func=calc_func, dim=num, pop=300, max_iter=1000, lb=[825, 30, 0, 30,0,0,0.16,0.37,0.002,0.003,0.01,0,30,0,  30, 30, 0.01, 0.01, 0.01],
              ub=[930, 540, 15, 140, 120, 0.5, 0.35, 0.51, 0.008, 0.012, 0.11, 0.18, 930, 30, 850, 160, 1.69, 0.23, 0.55], w=0.8, c1=0.2, c2=0.8)
    # pso = PSO(func=calc_func, dim=num, pop=500, max_iter=1000, lb=np.zeros(num), ub=np.ones(num),w=0.8, c1=0.2, c2=0.8)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

if __name__ == '__main__':
    features, targets = read_data()
    create_and_fit_model(features, targets)
    calc_pso()
