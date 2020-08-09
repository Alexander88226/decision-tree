import numpy as np
import pandas as pd

# number of data
n = 100

# header column
header_column = ["Tubular glands", "Exausted glands", "Cystic glands","Breakdown", 
                "Predicidua", "Variable vacuoles", "Rare Mitoses", "Surface breakdown", "Spindled stroma"]

# class labels
class_labels = ["Ovulatory breakdown (menstrual)",
            "Late Menstrual",
            "Hormonal therapy",
            "Anovulation (persistent follicle)",
            "Mid-cycle breakdown (follicle dysfunction)"]

# create random feature dataset
list_features = []
for index in range(n):
    feature_arr = np.random.choice([0, 1], size=(9,), p=[1./2, 1./2])
    list_features.append(feature_arr)

# create dataframe from dataset
feature_frame = pd.DataFrame(np.vstack(list_features))

# insert header column of dataframe
feature_frame.columns = header_column

# make boolean dataframe
feature_frame = feature_frame > 0

# create label column
label_arr = np.random.choice([0, 1, 2, 3, 4], size=(n, ), p=[1./5, 1./5, 1./5, 1./5, 1./5])
class_label_list = []
for label in label_arr:
    class_label_list.append(class_labels[label])
feature_frame["Phase"] = class_label_list

feature_frame.to_excel("samples.xlsx", index=False)