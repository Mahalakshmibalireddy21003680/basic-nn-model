# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

<img width="543" alt="Screenshot 2023-08-18 002357" src="https://github.com/Mahalakshmibalireddy21003680/basic-nn-model/assets/93427286/0ebaa9d1-fd79-458e-8753-301af633b539">



## DESIGN STEPS

<b>STEP 1:</b> Loading the dataset.

<b>STEP 2:</b> Split the dataset into training and testing.

<b>STEP 3:</b> Create MinMaxScalar objects ,fit the model and transform the data.

<b>STEP 4:</b> Build the Neural Network Model and compile the model.

<b>STEP 5:</b> Train the model with the training data.

<b>STEP 6:</b> Plot the performance plot.

<b>STEP 7:</b> Evaluate the model with the testing data.

## PROGRAM:
```
Developed By: B Mahalakshmi
RegNo: 212221240008
```

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('dl').sheet1
data = worksheet.get_all_values()
dataset = pd.DataFrame(data[1:], columns=data[0])
dataset = dataset.astype({'Input':'float'})
dataset = dataset.astype({'Output':'float'})
dataset.head()
X = dataset[['Input']].values
Y = dataset[['Output']].values
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.33,random_state = 20)
Scaler = MinMaxScaler()
Scaler.fit(x_train)
x_train_scale = Scaler.transform(x_train)
my_brain = Sequential([
    Dense(units = 4, activation = 'relu' , input_shape=[1]),
    Dense(units = 5),
    Dense(units = 1)

])
my_brain.compile(optimizer='rmsprop',loss='mse')
my_brain.fit(x=x_train_scale,y=y_train,epochs=20000)
loss_df = pd.DataFrame(my_brain.history.history)
loss_df.plot()
x_test1 = Scaler.transform(x_test)
my_brain.evaluate(x_test1,y_test)
X_n1 = [[30]]
input_scaled = Scaler.transform(X_n1)
my_brain.predict(input_scaled)
```
## Dataset Information


## OUTPUT:

### Training Loss Vs Iteration Plot

<img width="430" alt="Screenshot 2023-08-18 002534" src="https://github.com/Mahalakshmibalireddy21003680/basic-nn-model/assets/93427286/148d114f-43bb-4887-8b29-fc1ca67a565e">


### Test Data Root Mean Squared Error

<img width="433" alt="Screenshot 2023-08-18 002656" src="https://github.com/Mahalakshmibalireddy21003680/basic-nn-model/assets/93427286/19055b12-c2c7-40a2-978b-28c0c04ee327">


### New Sample Data Prediction

<img width="325" alt="Screenshot 2023-08-18 002720" src="https://github.com/Mahalakshmibalireddy21003680/basic-nn-model/assets/93427286/d7c95aa0-b84a-4e6f-9b9d-9fe583b62fdc">

## RESULT:
Therefore We successfully developed a neural network regression model for the given dataset.
