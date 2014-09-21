---
title: "Practical Machine Learning - Project 1"
output: html_document
---
# Practical Machine Learning - Project 1

## Synopsis

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, my goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here:[http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset). 

## Data Processing


```r
set.seed(1234)
#load required libraries
library(caret)
library(randomForest)
```

Firstly load the training and test data.


```r
trainDataFull <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
testData <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
```

To allow us to cross validate the model later we split the Training data into a training and validation set.


```r
inTrain <- createDataPartition(trainDataFull$classe, p=0.70, list=FALSE)
trainData <- trainDataFull[inTrain,]
validationData <- trainDataFull[-inTrain,]
```

An initial exploration of the data shows that there are 160 fields in our dataset and a number of fields that are not likely to be relevant to predicting the outcome (classe).  We want to remove as many of the irrelevant fields as possible to simplify our model without compromising accuracy.


```r
dim(trainData)
```

```
## [1] 13737   160
```

```r
head(trainData)
```

```
##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
## 7 7  carlitos           1323084232               368296 05/12/2011 11:23
##   new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
## 2         no         11      1.41       8.07    -94.4                3
## 3         no         11      1.42       8.07    -94.4                3
## 4         no         12      1.48       8.05    -94.4                3
## 5         no         12      1.48       8.07    -94.4                3
## 6         no         12      1.45       8.06    -94.4                3
## 7         no         12      1.42       8.09    -94.4                3
##   kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt
## 2                                                         
## 3                                                         
## 4                                                         
## 5                                                         
## 6                                                         
## 7                                                         
##   skewness_roll_belt skewness_roll_belt.1 skewness_yaw_belt max_roll_belt
## 2                                                                      NA
## 3                                                                      NA
## 4                                                                      NA
## 5                                                                      NA
## 6                                                                      NA
## 7                                                                      NA
##   max_picth_belt max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt
## 2             NA                         NA             NA             
## 3             NA                         NA             NA             
## 4             NA                         NA             NA             
## 5             NA                         NA             NA             
## 6             NA                         NA             NA             
## 7             NA                         NA             NA             
##   amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
## 2                  NA                   NA                   
## 3                  NA                   NA                   
## 4                  NA                   NA                   
## 5                  NA                   NA                   
## 6                  NA                   NA                   
## 7                  NA                   NA                   
##   var_total_accel_belt avg_roll_belt stddev_roll_belt var_roll_belt
## 2                   NA            NA               NA            NA
## 3                   NA            NA               NA            NA
## 4                   NA            NA               NA            NA
## 5                   NA            NA               NA            NA
## 6                   NA            NA               NA            NA
## 7                   NA            NA               NA            NA
##   avg_pitch_belt stddev_pitch_belt var_pitch_belt avg_yaw_belt
## 2             NA                NA             NA           NA
## 3             NA                NA             NA           NA
## 4             NA                NA             NA           NA
## 5             NA                NA             NA           NA
## 6             NA                NA             NA           NA
## 7             NA                NA             NA           NA
##   stddev_yaw_belt var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z
## 2              NA           NA         0.02         0.00        -0.02
## 3              NA           NA         0.00         0.00        -0.02
## 4              NA           NA         0.02         0.00        -0.03
## 5              NA           NA         0.02         0.02        -0.02
## 6              NA           NA         0.02         0.00        -0.02
## 7              NA           NA         0.02         0.00        -0.02
##   accel_belt_x accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y
## 2          -22            4           22            -7           608
## 3          -20            5           23            -2           600
## 4          -22            3           21            -6           604
## 5          -21            2           24            -6           600
## 6          -21            4           21             0           603
## 7          -22            3           21            -4           599
##   magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm
## 2          -311     -128      22.5    -161              34            NA
## 3          -305     -128      22.5    -161              34            NA
## 4          -310     -128      22.1    -161              34            NA
## 5          -302     -128      22.1    -161              34            NA
## 6          -312     -128      22.0    -161              34            NA
## 7          -311     -128      21.9    -161              34            NA
##   avg_roll_arm stddev_roll_arm var_roll_arm avg_pitch_arm stddev_pitch_arm
## 2           NA              NA           NA            NA               NA
## 3           NA              NA           NA            NA               NA
## 4           NA              NA           NA            NA               NA
## 5           NA              NA           NA            NA               NA
## 6           NA              NA           NA            NA               NA
## 7           NA              NA           NA            NA               NA
##   var_pitch_arm avg_yaw_arm stddev_yaw_arm var_yaw_arm gyros_arm_x
## 2            NA          NA             NA          NA        0.02
## 3            NA          NA             NA          NA        0.02
## 4            NA          NA             NA          NA        0.02
## 5            NA          NA             NA          NA        0.00
## 6            NA          NA             NA          NA        0.02
## 7            NA          NA             NA          NA        0.00
##   gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z magnet_arm_x
## 2       -0.02       -0.02        -290         110        -125         -369
## 3       -0.02       -0.02        -289         110        -126         -368
## 4       -0.03        0.02        -289         111        -123         -372
## 5       -0.03        0.00        -289         111        -123         -374
## 6       -0.03        0.00        -289         111        -122         -369
## 7       -0.03        0.00        -289         111        -125         -373
##   magnet_arm_y magnet_arm_z kurtosis_roll_arm kurtosis_picth_arm
## 2          337          513                                     
## 3          344          513                                     
## 4          344          512                                     
## 5          337          506                                     
## 6          342          513                                     
## 7          336          509                                     
##   kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
## 2                                                                       
## 3                                                                       
## 4                                                                       
## 5                                                                       
## 6                                                                       
## 7                                                                       
##   max_roll_arm max_picth_arm max_yaw_arm min_roll_arm min_pitch_arm
## 2           NA            NA          NA           NA            NA
## 3           NA            NA          NA           NA            NA
## 4           NA            NA          NA           NA            NA
## 5           NA            NA          NA           NA            NA
## 6           NA            NA          NA           NA            NA
## 7           NA            NA          NA           NA            NA
##   min_yaw_arm amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm
## 2          NA                 NA                  NA                NA
## 3          NA                 NA                  NA                NA
## 4          NA                 NA                  NA                NA
## 5          NA                 NA                  NA                NA
## 6          NA                 NA                  NA                NA
## 7          NA                 NA                  NA                NA
##   roll_dumbbell pitch_dumbbell yaw_dumbbell kurtosis_roll_dumbbell
## 2         13.13         -70.64       -84.71                       
## 3         12.85         -70.28       -85.14                       
## 4         13.43         -70.39       -84.87                       
## 5         13.38         -70.43       -84.85                       
## 6         13.38         -70.82       -84.47                       
## 7         13.13         -70.25       -85.10                       
##   kurtosis_picth_dumbbell kurtosis_yaw_dumbbell skewness_roll_dumbbell
## 2                                                                     
## 3                                                                     
## 4                                                                     
## 5                                                                     
## 6                                                                     
## 7                                                                     
##   skewness_pitch_dumbbell skewness_yaw_dumbbell max_roll_dumbbell
## 2                                                              NA
## 3                                                              NA
## 4                                                              NA
## 5                                                              NA
## 6                                                              NA
## 7                                                              NA
##   max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell min_pitch_dumbbell
## 2                 NA                                 NA                 NA
## 3                 NA                                 NA                 NA
## 4                 NA                                 NA                 NA
## 5                 NA                                 NA                 NA
## 6                 NA                                 NA                 NA
## 7                 NA                                 NA                 NA
##   min_yaw_dumbbell amplitude_roll_dumbbell amplitude_pitch_dumbbell
## 2                                       NA                       NA
## 3                                       NA                       NA
## 4                                       NA                       NA
## 5                                       NA                       NA
## 6                                       NA                       NA
## 7                                       NA                       NA
##   amplitude_yaw_dumbbell total_accel_dumbbell var_accel_dumbbell
## 2                                          37                 NA
## 3                                          37                 NA
## 4                                          37                 NA
## 5                                          37                 NA
## 6                                          37                 NA
## 7                                          37                 NA
##   avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell
## 2                NA                   NA                NA
## 3                NA                   NA                NA
## 4                NA                   NA                NA
## 5                NA                   NA                NA
## 6                NA                   NA                NA
## 7                NA                   NA                NA
##   avg_pitch_dumbbell stddev_pitch_dumbbell var_pitch_dumbbell
## 2                 NA                    NA                 NA
## 3                 NA                    NA                 NA
## 4                 NA                    NA                 NA
## 5                 NA                    NA                 NA
## 6                 NA                    NA                 NA
## 7                 NA                    NA                 NA
##   avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell gyros_dumbbell_x
## 2               NA                  NA               NA                0
## 3               NA                  NA               NA                0
## 4               NA                  NA               NA                0
## 5               NA                  NA               NA                0
## 6               NA                  NA               NA                0
## 7               NA                  NA               NA                0
##   gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y
## 2            -0.02             0.00             -233               47
## 3            -0.02             0.00             -232               46
## 4            -0.02            -0.02             -232               48
## 5            -0.02             0.00             -233               48
## 6            -0.02             0.00             -234               48
## 7            -0.02             0.00             -232               47
##   accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z
## 2             -269              -555               296               -64
## 3             -270              -561               298               -63
## 4             -269              -552               303               -60
## 5             -270              -554               292               -68
## 6             -269              -558               294               -66
## 7             -270              -551               295               -70
##   roll_forearm pitch_forearm yaw_forearm kurtosis_roll_forearm
## 2         28.3         -63.9        -153                      
## 3         28.3         -63.9        -152                      
## 4         28.1         -63.9        -152                      
## 5         28.0         -63.9        -152                      
## 6         27.9         -63.9        -152                      
## 7         27.9         -63.9        -152                      
##   kurtosis_picth_forearm kurtosis_yaw_forearm skewness_roll_forearm
## 2                                                                  
## 3                                                                  
## 4                                                                  
## 5                                                                  
## 6                                                                  
## 7                                                                  
##   skewness_pitch_forearm skewness_yaw_forearm max_roll_forearm
## 2                                                           NA
## 3                                                           NA
## 4                                                           NA
## 5                                                           NA
## 6                                                           NA
## 7                                                           NA
##   max_picth_forearm max_yaw_forearm min_roll_forearm min_pitch_forearm
## 2                NA                               NA                NA
## 3                NA                               NA                NA
## 4                NA                               NA                NA
## 5                NA                               NA                NA
## 6                NA                               NA                NA
## 7                NA                               NA                NA
##   min_yaw_forearm amplitude_roll_forearm amplitude_pitch_forearm
## 2                                     NA                      NA
## 3                                     NA                      NA
## 4                                     NA                      NA
## 5                                     NA                      NA
## 6                                     NA                      NA
## 7                                     NA                      NA
##   amplitude_yaw_forearm total_accel_forearm var_accel_forearm
## 2                                        36                NA
## 3                                        36                NA
## 4                                        36                NA
## 5                                        36                NA
## 6                                        36                NA
## 7                                        36                NA
##   avg_roll_forearm stddev_roll_forearm var_roll_forearm avg_pitch_forearm
## 2               NA                  NA               NA                NA
## 3               NA                  NA               NA                NA
## 4               NA                  NA               NA                NA
## 5               NA                  NA               NA                NA
## 6               NA                  NA               NA                NA
## 7               NA                  NA               NA                NA
##   stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
## 2                   NA                NA              NA
## 3                   NA                NA              NA
## 4                   NA                NA              NA
## 5                   NA                NA              NA
## 6                   NA                NA              NA
## 7                   NA                NA              NA
##   stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
## 2                 NA              NA            0.02            0.00
## 3                 NA              NA            0.03           -0.02
## 4                 NA              NA            0.02           -0.02
## 5                 NA              NA            0.02            0.00
## 6                 NA              NA            0.02           -0.02
## 7                 NA              NA            0.02            0.00
##   gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
## 2           -0.02             192             203            -216
## 3            0.00             196             204            -213
## 4            0.00             189             206            -214
## 5           -0.02             189             206            -214
## 6           -0.03             193             203            -215
## 7           -0.02             195             205            -215
##   magnet_forearm_x magnet_forearm_y magnet_forearm_z classe
## 2              -18              661              473      A
## 3              -18              658              469      A
## 4              -16              658              469      A
## 5              -17              655              473      A
## 6               -9              660              478      A
## 7              -18              659              470      A
```

We see from this that there are a number of columns that are unlikely to help with the modelling - eg X, user_name, cvtd_timestamp, raw_timestamp_part_1, raw_timestamp_part_2.  So we have removed these.

We will also remove any columns that show near zero variance as these are also unlikely to assist in building accurate predictions.


```r
trainData<-trainData[,-grep("X|user_name|cvtd_timestamp|raw_timestamp_part_1|raw_timestamp_part_2", names(trainData))]
nzv <- nearZeroVar(trainData)
trainData<-trainData[,-nzv]
```

Also from the initial investigation of the data there are a number of columns that are almost all NA.  These can also be removed to reduce the complexity of the model.


```r
t<-colSums(is.na(trainData))
print(t[t>0])
```

```
##            max_roll_belt           max_picth_belt            min_roll_belt 
##                    13453                    13453                    13453 
##           min_pitch_belt      amplitude_roll_belt     amplitude_pitch_belt 
##                    13453                    13453                    13453 
##     var_total_accel_belt            avg_roll_belt         stddev_roll_belt 
##                    13453                    13453                    13453 
##            var_roll_belt           avg_pitch_belt        stddev_pitch_belt 
##                    13453                    13453                    13453 
##           var_pitch_belt             avg_yaw_belt          stddev_yaw_belt 
##                    13453                    13453                    13453 
##             var_yaw_belt            var_accel_arm             max_roll_arm 
##                    13453                    13453                    13453 
##            max_picth_arm              max_yaw_arm             min_roll_arm 
##                    13453                    13453                    13453 
##            min_pitch_arm              min_yaw_arm      amplitude_pitch_arm 
##                    13453                    13453                    13453 
##        amplitude_yaw_arm        max_roll_dumbbell       max_picth_dumbbell 
##                    13453                    13453                    13453 
##        min_roll_dumbbell       min_pitch_dumbbell  amplitude_roll_dumbbell 
##                    13453                    13453                    13453 
## amplitude_pitch_dumbbell       var_accel_dumbbell        avg_roll_dumbbell 
##                    13453                    13453                    13453 
##     stddev_roll_dumbbell        var_roll_dumbbell       avg_pitch_dumbbell 
##                    13453                    13453                    13453 
##    stddev_pitch_dumbbell       var_pitch_dumbbell         avg_yaw_dumbbell 
##                    13453                    13453                    13453 
##      stddev_yaw_dumbbell         var_yaw_dumbbell        max_picth_forearm 
##                    13453                    13453                    13453 
##        min_pitch_forearm  amplitude_pitch_forearm        var_accel_forearm 
##                    13453                    13453                    13453
```

```r
trainData<-trainData[,colSums(is.na(trainData)) == 0]
```


## Model Creation
We have chosen to create a Random Forest model to predict because of their high level of accuracy.


```r
modFit<-randomForest(classe~.,data = trainData, prox = TRUE , importance=TRUE)
confusionMatrix(modFit$predicted, trainData$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3905    3    0    0    0
##          B    0 2652   10    0    0
##          C    0    3 2385   12    0
##          D    0    0    1 2239    5
##          E    1    0    0    1 2520
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.996, 0.998)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.997         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.998    0.995    0.994    0.998
## Specificity             1.000    0.999    0.999    0.999    1.000
## Pos Pred Value          0.999    0.996    0.994    0.997    0.999
## Neg Pred Value          1.000    0.999    0.999    0.999    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.163    0.183
## Detection Prevalence    0.284    0.194    0.175    0.163    0.184
## Balanced Accuracy       1.000    0.998    0.997    0.997    0.999
```

We see that we achieve a high level of accuracy with the confusion matrix showing in sample 99.7% accuracy.

## Cross-validation

Next we used our model to predict new values within the test set that we created for cross-validation.


```r
predcv<-predict(modFit, newdata=validationData)
confusionMatrix(predcv, validationData$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    3    0    0    0
##          B    0 1135    4    0    0
##          C    0    1 1020    4    0
##          D    0    0    2  960    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                         
##                Accuracy : 0.998         
##                  95% CI : (0.996, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.997         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.996    0.994    0.996    1.000
## Specificity             0.999    0.999    0.999    1.000    1.000
## Pos Pred Value          0.998    0.996    0.995    0.998    1.000
## Neg Pred Value          1.000    0.999    0.999    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.173    0.163    0.184
## Detection Prevalence    0.285    0.194    0.174    0.163    0.184
## Balanced Accuracy       1.000    0.998    0.997    0.998    1.000
```

```r
predRight<-sum(predcv== validationData$classe)
predWrong<-length(validationData$classe)-sum(predcv== validationData$classe)
```

The confusion matrix shows the predictions from our model were extremely accurate with only 14 misses out of the 5885 observations and an accuracy of 99.8%.  This gives us an expected out of sample error rate of 0.2%.


