# New-York-Taxi-Fare-Prediction

This is a project to predict the fare amount for a taxi ride in New York City given the pick-up and drop-off locations.

## Brief Introduction:

We have implemented four ways to solve this problem, using different models including Linear Regression, Neural Network, Gradiant Boost, Radom Forest.

## How to run programs:

First please go to https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data to download the dataset into 'src' folder.

```
git clone https://github.com/qzhizhou/New-York-Taxi-Fare-Prediction.git
cd New-York-Taxi-Fare-Prediction
pip install -r requirements.txt
cd src
jupyter notebook
```

Then you may explore each program and run them using the notebook.

## Best Result of each Model:
<p align="center">Linear Regression(5.75)</p>

![](https://github.com/qzhizhou/New-York-Taxi-Fare-Prediction/blob/master/pic/Linear.png)

<p align="center">Figure 1</p>
<p align="center">Xgboost(3.28)</p>

![](https://github.com/qzhizhou/New-York-Taxi-Fare-Prediction/blob/master/pic/xgboost.png)

<p align="center">Figure 2</p>
<p align="center">LightGBM(3.17)</p>

![](https://github.com/qzhizhou/New-York-Taxi-Fare-Prediction/blob/master/pic/lightGBM.png)

<p align="center">Figure 3</p>
<p align="center">Neural Network(3.56)</p>

![](https://github.com/qzhizhou/New-York-Taxi-Fare-Prediction/blob/master/pic/NeuralNetwork.png)

<p align="center">Figure 4</p>
