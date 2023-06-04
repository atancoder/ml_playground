# Classification
## Linear Decision Boundary
Linear logistic regression. But all clf_models were able to separate the data very well

<img width="562" alt="image" src="https://user-images.githubusercontent.com/10254642/236076600-2cc3ab68-a1dd-4de7-bb25-a5a30e009e0f.png">

## Rectangle Decision Boundary

**Best: Decision Tree**

Makes sense because that's how the data was constructed

<img width="582" alt="image" src="https://github.com/atancoder/ml_playground/assets/10254642/73fae7ab-9e41-43ff-9c63-25e7bc151689">

Quadratics approximate these quite well.
Polynomial logistic regression w/ degree=2

<img width="291" alt="image" src="https://user-images.githubusercontent.com/10254642/236064122-d777a8ea-6b6c-4e3c-8005-515cc65ccb67.png">

<img width="570" alt="image" src="https://user-images.githubusercontent.com/10254642/236093494-7a409be0-a245-4a5f-af9e-f1bd301a72ee.png">

## Triangular Decision Boundary

**Best: Neural Network w/ ReLu Activation**
```
model = tf.keras.Sequential(
      [
          tf.keras.layers.Dense(units=3, activation="relu"),
          tf.keras.layers.Dense(units=1, activation="sigmoid"),
      ]
  )
```
Ran over 300 epochs. You need a minimum of 3 units to fit this decision boundary. There are also some rare runs where NN gets stuck in a local minima and doesn't fit the data at all.
    
<img width="590" alt="image" src="https://github.com/atancoder/ml_playground/assets/10254642/0d900746-4710-4f7d-aca2-36afff4dc3bb">


Guassian logistic regression

<img width="578" alt="image" src="https://user-images.githubusercontent.com/10254642/236077022-fbff4c53-32f6-4d5c-8262-29c41a221fa4.png">

Quadratic generalizes better though to points outside the training set

<img width="577" alt="image" src="https://user-images.githubusercontent.com/10254642/236077243-dfb84944-e597-47f7-972c-73c613c1a929.png">

Deicison Trees don't fare well. Likely because they don't have any linear properties to imitate a slope

<img width="309" alt="image" src="https://github.com/atancoder/ml_playground/assets/10254642/07c6eb17-32e2-4a5f-9b98-d70f563bf768">


## Double Rectangle Decision Boundary
**Best: Decision Tree**

<img width="571" alt="image" src="https://github.com/atancoder/ml_playground/assets/10254642/63ac2f07-1def-4a59-b66a-ac9588396d0e">

Gaussian can also classify it but it was important to choose the right gamma for this to work. 

<img width="560" alt="image" src="https://user-images.githubusercontent.com/10254642/236089913-e0a8434b-5f0f-4511-a8c0-3151c37d8b9a.png">

No other models are able to capture the partitioned dataset

## Hyperbola Decision Boundary
**Best: Quadratic**

<img width="594" alt="image" src="https://user-images.githubusercontent.com/10254642/236091342-d67503ea-5ac9-4e7c-87c1-6e0f32bf03a4.png">

Gaussian can fits, but looks like it overfitted

<img width="536" alt="image" src="https://user-images.githubusercontent.com/10254642/236091381-4f974222-b55c-4e81-ab75-2e200d80c69c.png">

Decision trees can fit 


## Quadratic Decision Boundary W/ 1 Feature
1D feature where we positively classify if 3 <= x <= 8

We're able to get the following decision boundary with a polynomial kernel w/ degree = 2. Where y values > 0 => positive prediction.

![image](https://user-images.githubusercontent.com/10254642/236303035-3b019a4c-52aa-4928-8789-2f076c8cb67c.png)

This shows us that quadratic kernels are pretty good at classifying features that have a sweet spot (in this case, x=[3,8] was a sweet spot). 

<img width="563" alt="image" src="https://github.com/atancoder/ml_playground/assets/10254642/dec1170f-5fc7-4f57-a9cc-4cf112a103ae">


## Observations
Logistic Regression vs SVM performs very similarly. What's more important is how we model the data. Is it linear, polynomial, or do we use RBF? It seems like using RBF is a safe choice, as it fits the data very well in all shapes and sizes. The problem with using RBF is that it doesn't generalize well to data outside our training set range. It's also much harder to deduce the real pattern of the data (vs in a linear model, we can say what features most contribute to the classifier). 

StandardScaler was messing up accuracy in both quadratic models. The fix was on the data generation, making sure both features had the same 
range and that we used floats, instead of just integers
