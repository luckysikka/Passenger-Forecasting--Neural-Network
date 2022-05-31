# Bus-Service-Model- NeuralNet ,Support Vector Machine and Liner regression 

# Executive Summary

The objective of this project is to predict passengers for the existing and the new bus routes in the county of Surrey, England based on the features in the given database. The data was provided by the Surrey county council and consists of 37 bus operators from the period starting November 2015 to September 2018. 
 The database BusServices was imported into R through PGadmin using SQL queries. 
The data analysis has been done on mainly two tables: 

1. routes_daily_distinct: daily route information
2. routes_aggr_distinct: gives route information for the entire period from 2015 to 2018 aggregated by routes

Rows with missing values were observed in both tables in the dataset and consequently removed during data inspection. A correlation was then run to identify the relationship of the main variable, the passengers, with the other variables in the dataset.
The data visualization and prediction of passengers on the existing routes have been done on the ‘Routes daily’ table using RStudio & Power BI. The prediction of passengers on the existing bus routes was done using the popular time series analysis forecasting technique, ARIMA, which gave an accuracy of 48.2%.
For the prediction of passengers for the new routes, machine learning algorithms such as Linear Model, Linear Regression, Random Forest, Decision Tree, Neural Network and Linear Support Vector Machine (LSVM) were used on the ‘Route aggregator’ table in RStudio. The accuracy obtained from each of the algorithms was based on the values of RMSE and MAPE. These values were then compared across all the models and the model with the best accuracy was selected as the most precise in predicting passengers on the new bus routes.
At the end, to predict the passengers for the new bus routes, Neural Network model with a combination of training to test data of 80:20 is selected with RMSE of 1016.8 & MAPE of 99.4%. Therefore, the model will assist in efficient bus operation management & accurate, real-time and reliable passenger demand prediction. This will also aid in network planning, bus frequency settings, operation efficiency, costing and quality of service.


# Data Explanation and Preparation
  
The data for the Bus Services project is obtained from the Surrey county council from November 2015 to September 2018. There are a total of 37 bus operators and a total of 270 routes with distinct bus service numbers operating in Surrey. Out of these, 87 routes with different service numbers are analysed. The passenger numbers in the data set are the aggregated values & individual ticket data was used to calculate the aggregated values.
Surrey’s Bus Services map is created using 10 different tables:

* bus_route (bus routes)
* bus_stop (bus stops)
* train_stations (train stations)
* hospital (hospitals)
* households (delivery points)
* household_domestic_surrey (households)
* town - (town & village names) 
* surrey_boundary (border of Surrey)
* routes_daily_distinct (daily bus route information) 
* routes_aggr_distinct (bus route information over the entire time period) 

The ‘routes_daily_distinct’ and ‘routes_aggr_distinct’ were the main tables used for data analysis and the creation & selection of model with the least error to predict passengers for new and existing routes. 
The BusServices database was imported into R through PGadmin using SQL queries. Analysis was done using ‘passenger_avg’ as the target variable and ‘bus operator’, ‘route’, ‘weekday’, ‘peak’, ‘sdate’, ‘deviation’, ‘headway’, ‘households’, ‘hospitals’ and ‘train stations’ as the predictor or independent variables.
 
### Data Preparation and Cleaning 
When the data was summarised in SQL & R, some missing values were discovered in tables for the headway and deviation variables. Due to their limited number, the missing values were removed to avoid errors. 
After data cleansing, the routes_aggr_distinct and routes_daily_distinct tables were observed to have 244 distinct rows and 57,834 rows respectively.
There are a total of 12 modelling variables, including passengers, weekday, peak, deviation, household, train station, headway etc.
To establish and identify the relationship between the variables, correlation is carried out. The results of the correlation matrix are elucidated in the data visualisation section.
Passenger is the response variable in the analysis basis which all models are created. Therefore, for predicting the passengers on the existing route, Power BI is used on routes_daily_distinct table using the time series forecasting model Autoregressive Integrated Moving Average (ARIMA), in R.
Machine learning models like Linear model, Linear Regression, Decision tree, Random Tree, Neural Networks, Linear Support Vector Machine and Random forests are used for prediction for new routes in R.


### Forecasting 

![](Capture12.PNG)

Figure 12 demonstrates the forecasting of passengers for next 1 year based on the dataset provided from 2015 to 2018. The passenger forecasting has been made at 95% confidence interval and has been used to predict the passengers on existing bus routes. 

# Prediction and Classification Methods
 
Feature Selection: The most crucial step before deciding on the prediction or classification methods is the feature selection. It is one of the primary components and techniques of determining the most significant features to use as predictors in machine learning algorithms. Feature selection aims to identify a small subset of the significant features from the original set of features by removing unnecessary, redundant, or noisy variables as a dimensionality reduction technique. It generally leads to improved learning accuracy, improved learning performance, reduced cost of computation, and accurate model interpretability.

The feature selection for this project was done based on the relationship derived from correlation matrix of the variables and the objective of this project. The Surrey county council needs an idea of possible passenger footfall on each route at specific times of the day, to plan the buses in each route to avoid overcrowding. It is evident from the data visualization that more passengers travel on weekdays during peak hours. Therefore, the dataset was filtered for these two factors. 

It is observed that certain variables have high range (difference between the minimum & maximum value of the variable), as inferred from Table 1, may make the models biased towards the features having a high range. Therefore, log versions of passengers_avg, deviation_avg, households, headway_avg have been created to reduce the skewness of the variables and to normalize the range, so that the models can be free of any biases. 

The models were run on the log versions of the variables created. However, when the RMSE & MAPE values came too low, it indicated evidence of overfitting of the model. Therefore, log_passenger_avg was dropped. After the features were selected and data filtered, the final models were run on the variables ‘passengers_avg’, ‘log of headway_avg’, ‘log of deviation_avg’, ‘log of households’, ‘log of len’, ‘hospitals’, train_stations for weekdays and peak hours routes.


## Prediction Models - New bus routes

In this project, there are two objectives, one is to predict the passengers for the new bus routes, and the other to forecast passengers on the existing routes.

Based on the findings of the literature review, we have used six prediction models for predicting the passengers for the new bus routes. The dataset used for this is the ‘route_aggr_distinct’ table. The first step towards running a model is to divide the dataset into training and test data, so that the algorithm created on the training data can be tested on the test data. For this we have used two combinations, keeping the ratio of training to test data as 70:30 and 80:20. We have then run the six models on both combinations and the measured the accuracy of the models by two error metrics:
1.   Mean Absolute Percentage Error (MAPE) - most widely used measure for checking forecast accuracy. 
2. Root Mean Square Error (RMSE) - standard way to measure the error of a model in predicting quantitative data.
The lower values of RMSE & MAPE indicate better accuracy of the model and better predictions. 
The table below lists all the six models and their accuracy with both the combinations of training to test data. 


$$\mathrm{MSE}=\frac{1}{n} \sum_{k=1}^{n}\left(y_{k}-\hat{y_{k}}\right)^{2}$$





## Prediction Model - Existing bus routes

In order to forecast the passengers on the existing routes, we have created a new table where aggregation of passenger data is done, based on the date (sdate). For forecasting, Auto Regressive Integrated Moving Average ARIMA model is used, as it takes into consideration time series data, and creates forecasts based on the historic data obtained over time, as the name suggests moving average. The forecasting graph has been created using Power BI and is presented as figure 10. Passenger forecasting is an important aspect for bus operators to plan & provide timetable for bus operation management. The prediction can be supported by evaluating the mean absolute percentage error (MAPE) and the model has a fairly good forecast accuracy of 48.2%.   



Bus Service 
title: "R Notebook"

install.packages('tinytex')
tinytex::install_tinytex()
latexmk(engine = 'pdflatex', emulation = TRUE)
update.packages(ask = FALSE, checkBuilt = TRUE)
tinytex::tlmgr_update()
options(tinytex.verbose = TRUE)
library(tinytex)
install.packages('tinytex')
tinytex::install_tinytex()
if (!require("pacman")) install.packages("pacman")
pacman::p_load(knitr, captioner, bundesligR, stringr)



```{r}
#### Defining Function 
library(pacman); p_load(tidyverse)
source("C:/Users/oa01018/OneDrive - University of Surrey/Desktop/BusService/da_busservice/HumanNumbers.R")
rmse  <- function(y,yh) {sqrt(mean((y-yh)^2))}
mape  <- function(y,yh) {mean(abs((y-yh)/y))} # y=actual value, yh=modelled value (note issue when y is zero!)
frmse <- function(y,yh) {fmt(rmse(y,yh))}
fmape <- function(y,yh) {fmt(mape(y,yh)*100,'%')}
derr  <- function(y,yh,model="") {disp(model,'rmse=',frmse(y,yh),', mape=',fmape(y,yh))}
aplot <- function(y,yh, main="") {
  plot(y, yh, pch='.', cex=8, col = 'black', 
       ylab = "modelled", xlab = "observed",main=main); 
  abline(0,1, lwd=3, col='red');
}
                        
####  Connecting Group03 database from SQL 
library(DBI)
con <- DBI::dbConnect(odbc::odbc(), 
driver = "PostgreSQL ANSI(x64)", 
database = "group03", Uid = "group03", Pwd = "group03.", 
server = "cp-vms-ts1-01.surrey.ac.uk", port = 5432)
####  Fetching Table from SQL : ROUTES DAILY 
sql = "select distinct * from routes_daily"
File12 = dbGetQuery(con, sql);
View(File12)
saveRDS(File12, 'routes_daily.rds')
####  Fetching Table from SQL : ROUTES AGGR 
sql = "select distinct * from routes_aggr"
File2 = dbGetQuery(con, sql);
View(File2)
saveRDS(File2, 'routes_aggr.rds')
####  Library & Packages 
library(pacman); p_load(RPostgreSQL, DBI)
#install.packages("GGally")
library(GGally)
####  Checking of Missing Values 
length(which(is.na(File12))==TRUE)
length(which(is.na(File2))==TRUE)
####  Fixing Mising Values 
File2=na.omit(File2) 
# Checking of Missing Values after fixation 
length(which(is.na(File2))==TRUE)
####  checking correlation after removing the Geom & routes
File3 = File2 %>% dplyr::select(-c(geom,route))
 
Filecor = File12 %>% dplyr::select(-c(route,bus_operator,peak,weekday,sdate,bankholiday))
ggpairs(Filecor) 
####  Make Log Version of Headway Average
File3$logheadway_avg=log(File3$headway_avg)
####  Make Log Version of Deviation Average
File3$log_Deviation_avg=log(File3$deviation_avg)
####  Make Log Version of Passenger Average
File3$Log_Passengeravg=log(File3$passengers_avg)
library(pacman); p_load(tidyverse)
####  Feature selection A
File3A = subset(File3, select = -c(bus_operator,deviation_avg,headway_avg,passengers_avg,headway_sd,deviation_sd,passengers_sd) )
File3$log_household=log(File3$households)
File3$Log_length=log(File3$len)
####  Feature selection B
File3B = subset(File3, select = -c(bus_operator,deviation_avg,headway_avg,passengers_avg,headway_sd,deviation_sd,passengers_sd,households,len) )
ggpairs(File3B)
####  Feature selection & Filter of Peak & weekday  
File3C = File3 %>% filter(peak == "Peak", weekday == "Weekday")
FILE3D = File3C %>% dplyr::select(-c(bus_operator,weekday, peak,deviation_avg,headway_avg,Log_Passengeravg,headway_sd,deviation_sd,passengers_sd,households,len) )
####  Dividing the data in to Test (nr) and Train (nt)
nr = length(FILE3D$passengers_avg); nt=floor(.7*nr);
set.seed(123); train = sample(1:nr, nt);
#### Model 1 : Linear Model 
m1<- lm ( passengers_avg~., data = FILE3D[train,])
yh  = predict(m1, newdata=FILE3D[-train,])
y   = FILE3D$passengers_avg[-train]
derr(y,yh,"Linear Model: "); aplot(y,yh,"Linear Model")
summary(m1)
#### Model 2 : Linear Regression 
p_load(MASS)
ya = FILE3D$passengers_avg[train]
pX = FILE3D %>% dplyr::select(-c(passengers_avg))
X = as.matrix( pX[train,]) 
b = ginv(t(X) %*% X) %*% t(X) %*% ya
y = FILE3D$passengers_avg[-train]
Xt = as.matrix( pX[-train,]) 
yh = Xt %*% b
derr(y,yh,"Linear Regression: "); aplot(y,yh,"Linear Regression")
#### Model 3 : Random Forest Model 
p_load(randomForest)
set.seed(123);
rf = randomForest(formula = passengers_avg~., data = FILE3D, subset=train, importance = TRUE)
yh2  = predict(rf, newdata=FILE3D[-train,])
y2   = FILE3D$passengers_avg[-train]
derr(y2,yh2,"Random Forest : "); aplot(y2,yh2,"Random Forest")
I = importance(rf);I = I[order(-I[,1]),]
varImpPlot(rf)
#### Model 4 : Decision Tree 
p_load(tree)
dt = tree(passengers_avg~., data = FILE3D, subset=train)
yh  = predict(dt, newdata=FILE3D[-train,])
y   = FILE3D$passengers_avg[-train]
derr(y,yh,"Decision Tree : "); aplot(y,yh,"Decision Tree")
#### Model 5 : Neural Network 
p_load(neuralnet)
require(neuralnet)
nn=neuralnet(passengers_avg~.,data=FILE3D,hidden=1,act.fct="logistic",linear.output=FALSE)
yh  = predict(nn, newdata=FILE3D[-train,])
y   = FILE3D$passengers_avg[-train]
derr(y,yh,"Neural Network: "); aplot(y,yh,"Neural Network")
#### Model 6 : SVM Linear Model 
svmlinear = e1071::svm(formula = passengers_avg~., data = FILE3D[train,],
                   type = 'eps-regression', 
                   kernel = 'linear') 
yh  = predict(svmlinear, newdata=FILE3D[-train,])
y   = FILE3D$passengers_avg[-train]
derr(y,yh,"SVM Linear: "); aplot(y,yh,"SVM Linear")
#### Dividing the data in to Test (nr1) and Train (nt1) at 80%
nr1 = length(FILE3D$passengers_avg); nt1=floor(.8*nr1);
set.seed(123); train1 = sample(1:nr1, nt1);
#### Model 1 : Linear Model - 
m1<- lm ( passengers_avg~., data = FILE3D[train1,])
yh  = predict(m1, newdata=FILE3D[-train1,])
y   = FILE3D$passengers_avg[-train1]
derr(y,yh,"Linear Model: "); aplot(y,yh,"Linear Model")
summary(m1)
#### Model 2 : Linear Regression 
p_load(MASS)
ya = FILE3D$passengers_avg[train1]
pX = FILE3D %>% dplyr::select(-c(passengers_avg))
X = as.matrix( pX[train1,]) 
b = ginv(t(X) %*% X) %*% t(X) %*% ya
y = FILE3D$passengers_avg[-train1]
Xt = as.matrix( pX[-train1,]) 
yh = Xt %*% b
derr(y,yh,"Linear Regression: "); aplot(y,yh,"Linear Regression")
#### Model 3 : Random Forest Model 
p_load(randomForest)
set.seed(123);
rf = randomForest(formula = passengers_avg~., data = FILE3D, subset=train1, importance = TRUE)
yh2  = predict(rf, newdata=FILE3D[-train1,])
y2   = FILE3D$passengers_avg[-train1]

derr(y2,yh2,"Random Forest : "); aplot(y2,yh2,"Random Forest")
I = importance(rf);I = I[order(-I[,1]),]
varImpPlot(rf)
#### Model 4 : Decision Tree 
p_load(tree)
dt = tree(passengers_avg~., data = FILE3D, subset=train1)
yh  = predict(dt, newdata=FILE3D[-train1,])
y   = FILE3D$passengers_avg[-train1]
derr(y,yh,"Decision Tree : "); aplot(y,yh,"Decision Tree")

#### Model 5 : Neural Network 

p_load(neuralnet)
require(neuralnet)
nn1=neuralnet(passengers_avg~.,data=FILE3D,hidden=1,act.fct="logistic",linear.output=FALSE)
yh  = predict(nn1, newdata=FILE3D[-train1,])
y   = FILE3D$passengers_avg[-train1]
derr(y,yh,"Neural Network: "); aplot(y,yh,"Neural Network")

#### Model 6 : SVM Linear Model 
svmlinear = e1071::svm(formula = passengers_avg~., data = FILE3D[train,],
type = 'eps-regression', kernel = 'linear') 
yh  = predict(svmlinear, newdata=FILE3D[-train1,])
y   = FILE3D$passengers_avg[-train1]
derr(y,yh,"SVM Linear: "); aplot(y,yh,"SVM Linear")

#### Model 7 : Auto regressive integrated moving average (ARIMA MODEL) with grouping of Date and sum the passenger 
library(forecast)
GroupingArima<- File12 %>% group_by(sdate) %>% 
  summarise(TotalPassengers = sum(passengers))
GroupingArima<- ts(GroupingArima$TotalPassengers)
m8 <- auto.arima(GroupingArima)
y = GroupingArima
yh = m8$fitted
derr(y,yh,"Auto Regressive Integrated Moving Average Model: "); aplot(y,yh,"Auto Regressive Integrated Moving Average Model")
