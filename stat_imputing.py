from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler
import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix


class FeatureAnalysis(object):
    """
    Module to check the type of feature and automate the plots

    """

    def class_distribution(self,dataset):
        """
        Function returns the Dictionary of count of classes
        :param dataset: pandas columns
        :return: Dict
        """
        z = Counter(dataset)
        print(z)
        return z

    def regression_plots(self,dataset,count_features,feature_name):
        """
        Function to plot continious values
        :param dataset: dataframe
        :return: None
        """

        names = list(count_features.keys())
        values = list(count_features.values())

        fig, ax = plt.subplots()
        ax.scatter(names, values,color="black")
        ax.plot(names, values,color="red")

        ax.set(xlabel="NA", ylabel='NA',
               title=feature_name)
        ax.grid()

        fig.savefig(os.path.join(os.getcwd()+"/data/",feature_name+".png"))
        plt.show()

    def categorical_pie(self,dataset,count_features,feature_name):
        """
        Function to plot categorical values
        :param dataset:
        :return: None
        """

        names = list(count_features.keys())
        values = list(count_features.values())

        fig, ax = plt.subplots()
        ax.pie(values, labels=names, autopct='%1.1f%%', startangle=0)
        ax.axis('equal')
        plt.savefig(os.path.join(os.getcwd()+"/data/",feature_name+ ".png"))
        plt.show()

    def categorical_bar(self,dataset,count_features,feature_name):
        """
        Function to plot categorical features
        :param dataset:
        :return: None
        """

        names = list(count_features.keys())
        values = list(count_features.values())
        y_pos = np.arange(min(names), max(names) + 1)

        plt.rcdefaults()
        fig, ax = plt.subplots()
        ax.barh(names, values, color='blue', align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel("COUNTS")
        ax.set_title(feature_name)

        plt.savefig(os.path.join(os.getcwd()+"/data/",feature_name+".png"))
        plt.show()

    def choose_plot(self,dataset):
        """
        Function to choose the type of plot
        :param dataset:
        :return:
        """

        for i in dataset:
            count_features = self.class_distribution(dataset[i])
            if len(count_features.keys()) > 10:
                self.regression_plots(dataset[i],count_features,i)
            else:
                self.categorical_bar(dataset[i],count_features,i)
                self.categorical_pie(dataset[i],count_features,i)


class ReadSplit(object):

    def read_data(self, path):
        """
        	Read data from CSV file
        	:param path: string
        	:return: pandas dataframe
        """

        try:
            dataset = pd.read_csv(path, low_memory=False)
            return dataset
        except Exception as e:
            print(e)

    def undersampling(self, dataframe, n):
        """
        	Function returns randomly selected rows from a dataframe

        	:param dataframe: pandas dataframe
        	:param n: n number of rows
        	:return: pandas dataframe
        """

        c = list(range(0, len(dataframe)))
        sample_rows = random.sample(c, n)

        new_data = pd.DataFrame()
        for i in sample_rows:
            new_data = new_data.append(dataframe.iloc[i])

        print("new_data_rows", len(new_data))
        return new_data

    def oversampling(self, dataset, n):
        """
               Function returns randomly selected rows from a dataframe

               :param dataframe: pandas dataframe
               :param n: n number of rows
               :return: pandas dataframe
               """

        initial_copy = dataset.copy(deep=True)
        try:
            while (len(dataset) < n):
                dataset = dataset.append(initial_copy)
            return dataset
        except Exception as e:
            print(e)

    def synthetic_sampling_SMOTE(self, dataset):
        """
        Function to generate synthetic samples
            :param dataset:
            :return:
        """
        try:

            data = dataset.iloc[:, :-2]
            y = dataset.iloc[:, -1]
            X_resampled, y_resampled = SMOTE().fit_sample(data, y)
            X_resampled = pd.DataFrame(X_resampled)
            y_resampled = pd.DataFrame(y_resampled)
            new_dataset = pd.concat([X_resampled, y_resampled], axis=1)
            return new_dataset
        except Exception as e:
            print(e)

    def synthetic_sampling_ADASYN(self, dataset):
        """

            :param dataset:
            :return:
        """
        try:
            data = dataset.iloc[:, :-2]
            y = dataset.iloc[:, -1]
            X_resampled, y_resampled = ADASYN().fit_sample(data, y)
            X_resampled = pd.DataFrame(X_resampled)
            y_resampled = pd.DataFrame(y_resampled)
            new_dataset = pd.concat([X_resampled, y_resampled], axis=1)
            return new_dataset
        except Exception as e:
            print(e)

    def random_oversampler(self, dataset):
        """

            :param dataset:
            :return:
        """
        try:

            data = dataset.iloc[:, :-2]
            y = dataset.iloc[:, -1]
            X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_sample(data, y)
            X_resampled = pd.DataFrame(X_resampled)
            y_resampled = pd.DataFrame(y_resampled)
            new_dataset = pd.concat([X_resampled, y_resampled], axis=1)
            return new_dataset
        except Exception as e:
            print(e)

    def split_dataset(self, dataset, sampling_type):
        """
            Function to split datset into train & test sets
            :param dataset: pandas dataframe
            :return: train/test split data
        """

        try:

            y = dataset.iloc[:, -1]
            count_predictors = Counter(y)
            #print(count_predictors)

            if count_predictors[0] < 0.5 * count_predictors[1] or count_predictors[0] > 0.5 * count_predictors[1]:
                if sampling_type == 1:
                    print(" Initiating random oversampling")
                    dataset = self.random_oversampler(dataset)

                elif sampling_type == 2:
                    print(" Initiating synthetic ADASYN")
                    dataset = self.synthetic_sampling_ADASYN(dataset)

                elif sampling_type == 3:
                    print(" Initiating synthetic SMOTE")
                    dataset = self.synthetic_sampling_SMOTE(dataset)

            data = dataset.iloc[:, :-2]
            y = dataset.iloc[:, -1]
            #count_predictors = Counter(y)
            #print(count_predictors)

            data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
            data = data.replace({np.nan: 0})
            data = data.fillna(0)
            train_input, test_input, train_output, test_output = train_test_split(data, y, test_size=0.2,
                                                                                  shuffle=True)
            return train_input, test_input, train_output, test_output
        except Exception as e:
            print(e)
            return None,None,None,None


class PerfMetric(object):
    """
    Module comprising performance metrics for classification and Regression
    """

    def confusion_matrix(self,actual_output,predicted_output):
            """
            Function to Compute Precesion, Accuracy, Sensitivity, Specificity

            :param predicted_output:
            :param actual_output:
            :return: Precesion, Accuracy, Sensitivity, Specificity
            """

            conf_matrix = confusion_matrix(actual_output,predicted_output)
            true_p = conf_matrix[0][0]
            true_n = conf_matrix[1][1]
            false_p =  conf_matrix[0][1]
            false_n  = conf_matrix[1][0]

            # Sensitivity = TP /(TP + FN).Maximize Sensitivity/Penalize False N,eg in Fraud Detection,Cancer Prediction.
            sensitivity = true_p / (true_p + false_n)

            # Specificity = TN/(TN+FP) Penalize FP/Maximize specificity in case of phishing filter..
            specificity = true_n/(true_n+false_p)

            precesion = true_p/ (true_n+true_p)

            # Use accuracy metric when classes are balanced
            accuracy = true_p / (true_p + true_n + false_p + false_n)

            # F1_score is the harmonic mean on Precesion & Recall
            f1_score = (2 * precesion * sensitivity) / (precesion + sensitivity)

            return precesion, accuracy, f1_score, sensitivity, specificity

    def log_loss(self,actual_output, proba_output):
            """
            Function to compute log loss

            log loss = (-1/N) * (  [ 1~N Samples ] [ 1~M Classes ] Actual_ij * Proba_ij )


            :return logloss float
            """
            return log_loss(actual_output,proba_output)


if __name__ == "__main__":
    obj = ReadSplit()
