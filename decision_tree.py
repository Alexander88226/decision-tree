from PyQt5 import QtWidgets, uic, QtCore, QtGui

from PyQt5.QtWidgets import QFileDialog, QMessageBox

import sys
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder # Import Ordinal Encoder

def display_error_message(content):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("Error")
    msg.setInformativeText(content)
    msg.setWindowTitle("Error")
    msg.exec_()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        uic.loadUi("main.ui", self)
        # col_header = ["", ""]
        self.dataset_file_path = ""
        self.Tubular_glands = False
        self.Exausted_glands = False
        self.Cystic_glands = False
        self.Breakdown = False
        self.Predicidua = False
        self.Variable_vacuoles = False
        self.Rare_Mitoses = False
        self.Surface_breakdown = False
        self.Spindled_stroma = False
        self.clf = DecisionTreeClassifier()
        self.label_encoder1 = LabelEncoder()
        self.label_encoder2 = LabelEncoder()
        self.pre_trained = False
    @QtCore.pyqtSlot()
    def browse(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Excel Files (*.xlsx)", options=options)
        if fileName:
            self.dataset_file_path = fileName
            self.dataset_file.setText(fileName)

    @QtCore.pyqtSlot()
    def evaluate(self):
        
        if not self.pre_trained:
            display_error_message("Please train the model with dataset")
            return
        self.Tubular_glands = self.feature1.isChecked()
        self.Exausted_glands = self.feature2.isChecked()
        self.Cystic_glands = self.feature3.isChecked()
        self.Breakdown = self.feature4.isChecked()
        self.Predicidua = self.feature5.isChecked()
        self.Variable_vacuoles = self.feature6.isChecked()
        self.Rare_Mitoses = self.feature7.isChecked()
        self.Surface_breakdown = self.feature8.isChecked()
        self.Spindled_stroma = self.feature9.isChecked()
        
        X = [
            self.Tubular_glands,
            self.Exausted_glands,
            self.Cystic_glands,
            self.Breakdown,
            self.Predicidua,
            self.Variable_vacuoles,
            self.Rare_Mitoses,
            self.Surface_breakdown,
            self.Spindled_stroma
            ]
        X = self.label_encoder1.fit_transform(X)
        predictions = self.clf.predict([X])
        
        phase = self.label_encoder2.inverse_transform(predictions)
        self.result_phase.setText(phase.item(0))

    @QtCore.pyqtSlot()
    def train(self):

        if self.dataset_file_path == "":
            display_error_message('Please Select Dataset File')
            return
        

        self.pre_trained = False
        # Load Data
        data = pd.read_excel(self.dataset_file_path, index_col=None)

        # check if dataset is valid
        data_columns = data.columns
        print(len(data_columns))
        if len(data_columns) != 10:
            display_error_message('Invalid Dataset File')
        # Label Encoding
        data1 = data.copy()
        self.label_encoder1 = LabelEncoder()
        self.label_encoder2 = LabelEncoder()
        for column in data_columns[:-1]:
            data1[column] = self.label_encoder1.fit_transform(data[column])
        data1[data_columns[-1]] = self.label_encoder2.fit_transform(data[data_columns[-1]])

        # Feature Selection
        X = data1[data_columns[:-1]] # Features
        y = data1[data_columns[-1]] # Target variable

        unique_label = list(data[data_columns[-1]].unique())
        sorted(unique_label)

        # Splitting Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Building Decision Tree Model
        # Create Decision Tree classifer object
        self.clf = DecisionTreeClassifier(criterion="entropy")

        # Train Decision Tree Classifer
        self.clf = self.clf.fit(X_train,y_train)

        #Predict the response for test dataset
        y_pred = self.clf.predict(X_test)

        # Evaluating Model
        print("Accuracy:" ,metrics.accuracy_score(y_test, y_pred))

        self.pre_trained = True


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
