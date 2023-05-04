# ml_playground
## Linear Decision Boundary

<img width="562" alt="image" src="https://user-images.githubusercontent.com/10254642/236076600-2cc3ab68-a1dd-4de7-bb25-a5a30e009e0f.png">
Linear logistic regression

## Circular Decision Boundary

<img width="291" alt="image" src="https://user-images.githubusercontent.com/10254642/236064122-d777a8ea-6b6c-4e3c-8005-515cc65ccb67.png">
Polynomial logistic regression w/ degree=2


## Triangular Decision Boundary

<img width="578" alt="image" src="https://user-images.githubusercontent.com/10254642/236077022-fbff4c53-32f6-4d5c-8262-29c41a221fa4.png">
Guassian logistic regression

Quadratic generalizes better though to points outside the training set
<img width="577" alt="image" src="https://user-images.githubusercontent.com/10254642/236077243-dfb84944-e597-47f7-972c-73c613c1a929.png">

## Double Circle Decision Boundary
<img width="560" alt="image" src="https://user-images.githubusercontent.com/10254642/236089913-e0a8434b-5f0f-4511-a8c0-3151c37d8b9a.png">
Gaussian was the best one that can classify. It was important to choose the right gamma for this to work. 

## Observations
Logistic Regression vs SVM performs very similarly. What's more important is how we model the data. Is it linear, polynomial, or do we use RBF? It seems like using RBF is a safe choice, as it fits the data very well in all shapes and sizes. The problem with using RBF is that it doesn't generalize well to data outside our training set range. It's also much harder to deduce the real pattern of the data (vs in a linear model, we can say what features most contribute to the classifier). 

StandardScaler was messing up accuracy in both quadratic models. The fix was on the data generation, making sure both features had the same 
range and that we used floats, instead of just integers
