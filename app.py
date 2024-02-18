import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class App:

    def __init__(self):

        st.toast("Loading. Please wait...")
        time.sleep(1)
        st.title("Cezeri-Baykar YZUP Modül Projesi")
        st.write("""
                 ### Author: Muhammet Ali Şentürk
                 """)
        st.write("""
                ##### Measure the performance of different classifiers on the dataset you chose
                """)
        self.target_variable = None
        self.useless_columns = []
        self.x_col_to_plot, self.y_col_to_plot = None, None
        self.classifier_name = None
        self.params = {}
        self.model = None
        self.data = None
        self.X, self.y = None, None

    def run(self):

        self.generate()

    def init_streamlit_page(self):

        self.target_variable = st.sidebar.selectbox(
            "Select target variable",
            sorted(set(self.data.columns)),
            index=13
        )
        self.useless_columns = st.sidebar.multiselect(
            "Select unnecessary columns to drop",
            sorted(set(self.data.columns)),
            default=["Unnamed: 32", "id"]
        )
        self.x_col_to_plot = st.sidebar.selectbox(
            "Select variable that will lie on X axis in scatter plot",
            sorted(set(self.data.columns)),
            index=21
        )
        self.y_col_to_plot = st.sidebar.selectbox(
            "Select variable that will lie on Y axis in scatter plot",
            sorted(set(self.data.columns)),
            index=30
        )
        self.classifier_name = st.sidebar.selectbox(
            "Select classifier",
            ("KNN", "SVM", "Naive-Bayes")
        )

    def preprocess(self):

        self.data.drop(self.useless_columns, axis=1, inplace=True)
        self.data[self.target_variable] = self.data[self.target_variable].apply(lambda x: 1 if x == "M" else 0)
        st.write("Last 10 entries of the data after preprocessing")
        st.table(self.data.tail(10))
        self.X = self.data.drop(self.target_variable, axis=1)
        self.y = self.data[self.target_variable]
        self.plot_corr_matrix()
        self.plot_scatter()

    def plot_corr_matrix(self):

        fig = plt.figure()
        sns.heatmap(self.data.corr(), annot=False, cmap="YlOrRd", fmt=".2f", mask=np.triu(self.data.corr()))
        return fig

    def plot_scatter(self):

        fig = plt.figure()
        data_m = self.data[self.data[self.target_variable] == 1]
        data_b = self.data[self.data[self.target_variable] == 0]

        plt.scatter(data_m[self.x_col_to_plot], data_m[self.y_col_to_plot],
                    color="Red", alpha=0.3, edgecolors="r", linewidths=1)
        plt.scatter(data_b[self.x_col_to_plot], data_b[self.y_col_to_plot],
                    color="Green", alpha=0.3, edgecolors="g", linewidths=1)

        plt.xlabel(self.x_col_to_plot)
        plt.ylabel(self.y_col_to_plot)
        plt.legend(["kotu", "iyi"])

        return fig

    def get_classifier(self, best_params):

        if self.classifier_name == "SVM":
            self.model = SVC(**best_params)
            self.params = {"C": [0.1, 1, 10]}
        elif self.classifier_name == "KNN":
            self.model = KNeighborsClassifier(**best_params)
            self.params = {"n_neighbors": np.arange(1, 11, 1)}
        else:
            self.model = GaussianNB()

    def deploy_best_model(self, xtr, xts, ytr):        # Includes normalization and grid-search

        scaler = StandardScaler()
        s_X_train = scaler.fit_transform(xtr)
        s_X_test = scaler.transform(xts)

        cv_model = GridSearchCV(self.model, self.params, cv=5)
        cv_model.fit(s_X_train, ytr)

        self.get_classifier(cv_model.best_params_)
        tuned_model = self.model
        tuned_model.fit(s_X_train, ytr)
        preds = tuned_model.predict(s_X_test)

        return preds

    def calculate_scores(self, true, pred):

        accuracy = round(accuracy_score(true, pred), 2)
        precision = round(precision_score(true, pred), 2)
        recall = round(recall_score(true, pred), 2)
        f1score = round(f1_score(true, pred), 2)

        st.write(f"### Classifier = {self.classifier_name}")
        st.write(f"Accuracy: %", accuracy)
        st.write(f"Precision: %", precision)
        st.write(f"Recall: %", recall)
        st.write(f"F1 Score: %", f1score)

    def plot_confusion_matrix(self, true, pred):

        fig = plt.figure()
        sns.heatmap(confusion_matrix(true, pred), annot=True, cmap="Blues")

        plt.title(f"Confusion Matrix of {self.classifier_name} Model", fontsize=15, pad=15, fontweight="bold")
        plt.xlabel("y_pred")
        plt.xticks([0.5, 1.5], ["iyi", "kotu"])
        plt.ylabel("y_true")
        plt.yticks([0.5, 1.5], ["iyi", "kotu"])

        return fig

    def generate(self):

        uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:
            self.data = pd.read_csv(uploaded_file)
            self.init_streamlit_page()
            st.write("First look to the data")
            st.write(self.data.head(10))
            self.X = self.data.drop(self.target_variable, axis=1)
            self.y = self.data[self.target_variable]
            self.preprocess()

            with st.spinner("Plotting the correlation heatmap"):
                time.sleep(1)
                st.pyplot(self.plot_corr_matrix())
            with st.spinner("Plotting the scatter plot"):
                time.sleep(1)
                st.pyplot(self.plot_scatter())
            self.get_classifier(dict())

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)

            model_preds = self.deploy_best_model(X_train, X_test, y_train)
            self.calculate_scores(y_test, model_preds)
            with st.spinner("Plotting the confusion matrix"):
                time.sleep(1)
                st.pyplot(self.plot_confusion_matrix(y_test, model_preds))
