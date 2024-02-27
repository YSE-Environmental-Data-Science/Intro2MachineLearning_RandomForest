# Introduction to Machine Learning with RandomForest

### Install Packages:
```{r, include=T}
install.packages("randomForest")
install.packages("tidyverse")
install.packages("GGally")
```
### Load Libraries:
```{r, include=T}
library(randomForest)
library(tidyverse)
library(GGally)
```
Boosted regression trees (BRT) represent a versatile machine learning technique applicable to both classification and regression tasks. This approach facilitates the assessment of the relative significance of numerous variables associated with a target response variable. In this workshop, our focus will be on utilizing BRT to develop a model for monthly methane fluxes originating from natural ecosystems. We'll leverage climate and moisture conditions within these ecosystems to enhance predictive accuracy and understanding.

Read in the data:
```{r, include=T}
load('RANDOMFOREST_DATASET.RDATA' )

```

Our ultimate interest is in predicting monthly methane fluxes using both dynamic and static attribute of ecosystems. Before we start modeling with the data, it is a good practice to first visualize the variables. The ggpairs() function from the GGally package is a useful tool that visualizes the distribution and correlation between variables:

```{r, include=T}
ggpairs(fluxnet, columns = c(3:7, 12:13))
```
Next we need to divide the data into testing (20%) and training (80%) sets in a reproducible way:
```{r, include=T}
set.seed(111) # set the randomnumber generator

#create ID column
fluxnet$id <- 1:nrow(fluxnet)

#use 80% of dataset as training set and 30% as test set 
train <- fluxnet %>% dplyr::sample_frac(0.80)
test  <- dplyr::anti_join(fluxnet, train, by = 'id')
```
We will use the randomForest() function to predict monthly natural methane efflux using several variables in the dataset. A few other key statements to use in the randomForest() function are:

1. keep.forest = T: This will save the random forest output, which will be helpful in summarizing the results.
2. importance = TRUE: This will assess the importance of each of the predictors, essential output in random forests.
3. mtry = 1: This tells the function to randomly sample one variable at each split in the random forest. For applications in regression, the default value is the number of predictor variables divided by three (and rounded down). In the modeling, several small samples of the entire data set are taken. Any observations that are not taken are called “out-of-bag” samples.
4. ntree = 500: This tells the function to grow 500 trees. Generally, a larger number of trees will produce more stable estimates. However, increasing the number of trees needs to be done with consideration of time and memory issues when dealing with large data sets.

Our response variable in the random forests model is FCH4_F_gC and predictors are P_F, TA_F, VPD_F, IGBP, NDVI, and EVI. We will only explore a few of these variables below:

```{r, include=T}
FCH4_F_gC.rf <- randomForest(FCH4_F_gC ~ P_F + TA_F + VPD_F ,
                        data = train,
                        keep.forest = T,
                        importance = TRUE, 
                        mtry = 1,
                        ntree = 500,
                        keep.inbag=TRUE)
FCH4_F_gC.rf
```
Note the mean of squared residuals and the percent variation explained (analogous to R-squared) provided in the output.

Visualize the out-of-bag error rates of the random forests models using the plot() function. In this application, although we specified 500 trees, the out-of-bag error generally stabilizes after 100 trees:

```{r, include=T}
plot(FCH4_F_gC.rf)
```

Some of the most helpful output in random forests is the importance of each of the predictor variables. The importance score is calculated by evaluating the regression tree with and without that variable. When evaluating the regression tree, the mean square error (MSE) will go up, down, or stay the same.

If the percent increase in MSE after removing the variable is large, it indicates an important variable. If the percent increase in MSE after removing the variable is small, it’s less important.

The importance() function prints the importance scores for each variable and the varImpPlot() function plots them:
```{r, include=T}
importance(FCH4_F_gC.rf)

varImpPlot(FCH4_F_gC.rf)
```
Another aspect of model evaluation is comparing predictions. Although random forests models are often considered a “black box” method because their results are not easily interpreted, the predict() function provides predictions of total tree mass:

```{r, include=T}
  train$PRED.TPVPD <- predict(FCH4_F_gC.rf, train)
```
Compare the observed (FCH4_F_gC) versus predicted (PRED.TPVPD):

```{r, include=T}
  ggplot() + geom_point( data = train, aes( x=FCH4_F_gC, y= PRED.TPVPD )) +
  geom_smooth(method='lm')
  
  summary(lm(data=train,  PRED.TPVPD~FCH4_F_gC))
  
```
See how well the model performs on data that was not used to train the model:

```{r, include=T}
  test$PRED.TPVPD <- predict(FCH4_F_gC.rf, test)
  
  ggplot() + geom_point( data = test, aes( x=FCH4_F_gC, y= PRED.TPVPD )) +
  geom_smooth(method='lm')
  
  summary(lm(data=test,  PRED.TPVPD~FCH4_F_gC))
```
# Final Model Development:
The current model includes only climate variables from the tower. Use either a forward or backward selection method to develop your final model using your own data sets.

The forward selection approach starts with no variables and adds each new variable incrementally, testing for statistical significance, while the backward elimination method begins with a full model and then removes the least statistically significant variables one at a time.

Save your final model and datasets in a .Rdata object for next class where we will perform sensitivity analyses on the models. 


```{r, include=T}
save( FCH4_F_gC.rf , file="FinalModel.RDATA")
```
