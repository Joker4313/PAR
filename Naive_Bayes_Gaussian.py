import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import Data_Input

x_train = Data_Input.return_X_train_Array()
y_train = Data_Input.return_y_train_Array()

x_test = Data_Input.return_X_test_Array()
y_test = Data_Input.return_y_test_Array()



model = GaussianNB()
model.fit(x_train, y_train)

print("Model Score: ", model.score(x_test, y_test))
print(model.predict(x_test))