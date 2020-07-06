
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.svm import SVR
from scipy.cluster import  hierarchy
import numpy as np
import warnings
warnings.simplefilter("ignore")
from impyute.imputation.cs import mice


class Unsup:
    """
    Models building data points with unsupervised learning
    """
    def _init__(self,data):
        self.data=data
        pass

    def cluster(self):        
        threshold = 0.1
        Z = hierarchy.linkage(self.data,"average", metric="cosine")
        C = hierarchy.fcluster(Z, threshold, criterion="distance")
    
    def MICE(self):
        # start the MICE training
        imputed_training=mice(self.data)

class Cat():
    """
    
    Models with multi input and multi output classifier
    """

    def __init__(self,data_x,data_y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_x, data_y,
                                                    train_size=0.75,
                                                    random_state=4)
 
    def SVM(self):
        """
        """
        class_multisvm = MultiOutputClassifier(SVM(kernel='rbf'))

        # Fit on the train data
        class_multisvm.fit(self.X_train, self.y_train)

        # Check the prediction score
        score = class_multisvm.score(self.X_test, self.y_test)
        print("The prediction score on the test data is {:.2f}%".format(score*100))                                                 

    def RF(self):
        
        class_multirf = MultiOutputClassifier(RandomForestClassifier(max_depth=30,
                                                                random_state=0))

        # Fit on the train data
        class_multirf.fit(self.X_train, self.y_train)

        # Check the prediction score
        score = class_multirf.score(self.X_test, self.y_test)

        print("The prediction score on the test data is {:.2f}%".format(score*100))

    def NN(self):
        """
        Neural Net Model
        """
        
        N = len(self.X_train)
        dim_x=self.X_train.shape[1]
        dim_y=self.y_train.shape[1]
        
        XX_train = self.X_train
        XX_test = self.X_test

        YY_train_labels = self.y_train
        YY_test_labels = self.y_test

        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(np.concatenate((XX_train, XX_test)))

        X_train_scale = min_max_scaler.transform(XX_train)
        X_test_scale = min_max_scaler.transform(XX_test)

        print(X_train_scale.shape,X_test_scale.shape )

        model = Sequential()
        model.add(Dense(input_dim=dim_x, output_dim=60))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(input_dim=60, output_dim=80))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(input_dim=80, output_dim=60))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(input_dim=60, output_dim=40))
        model.add(Activation('relu'))
        model.add(Dense(input_dim=400, output_dim=20))
        model.add(Activation('relu'))
        model.add(Dense(input_dim=20, output_dim=dim_y))
        model.add(Activation('sigmoid'))

        model.compile(loss='mean_squared_error', optimizer='rmsprop')

        model.fit(X_train_scale, YY_train_labels,
                batch_size=2, epochs=10,
                verbose=2,
                validation_data=(X_test_scale, YY_test_labels))

        print(model.predict(X_train_scale, batch_size=1))

        return model    
class Cont:
    """
    
    Models with multi input and multi output regression
    """
    def __init__(self,data_x,data_y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_x, data_y,
                                                    train_size=0.75,
                                                    random_state=4)

    def SVR(self):
        """
        SVR Model
        """
        regr_multirf = MultiOutputRegressor(SVR(kernel='rbf'))

        # Fit on the train data
        regr_multirf.fit(self.X_train, self.y_train)

        # Check the prediction score
        score = regr_multirf.score(self.X_test, self.y_test)
        print("The prediction score on the test data is {:.2f}%".format(score*100))                                                 

    def RF(self):
        """
        Random Forest Model
        """
        
        regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=30,
                                                                random_state=0))

        # Fit on the train data
        regr_multirf.fit(self.X_train, self.y_train)

        # Check the prediction score
        score = regr_multirf.score(self.X_test, self.y_test)
        print("The prediction score on the test data is {:.2f}%".format(score*100))


    def NN(self):
        """
        Neural Net Model
        """
        
        N = len(self.X_train)
        dim_x=self.X_train.shape[1]
        dim_y=self.y_train.shape[1]
        
        XX_train = self.X_train
        XX_test = self.X_test

        YY_train_labels = self.y_train
        YY_test_labels = self.y_test

        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(np.concatenate((XX_train, XX_test)))

        X_train_scale = min_max_scaler.transform(XX_train)
        X_test_scale = min_max_scaler.transform(XX_test)

        print(X_train_scale.shape,X_test_scale.shape )

        model = Sequential()
        model.add(Dense(input_dim=dim_x, output_dim=60))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(input_dim=60, output_dim=80))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(input_dim=80, output_dim=60))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(input_dim=60, output_dim=40))
        model.add(Activation('relu'))
        model.add(Dense(input_dim=400, output_dim=20))
        model.add(Activation('relu'))
        model.add(Dense(input_dim=20, output_dim=dim_y))
        model.add(Activation('sigmoid'))

        model.compile(loss='mean_squared_error', optimizer='rmsprop')

        model.fit(X_train_scale, YY_train_labels,
                batch_size=2, epochs=10,
                verbose=2,
                validation_data=(X_test_scale, YY_test_labels))

        print(model.predict(X_train_scale, batch_size=1))

        return model