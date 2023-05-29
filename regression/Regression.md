# Regression 

## Linear Regression
Linear, Quadratic, and Gaussian work equally well

<img width="565" alt="image" src="https://user-images.githubusercontent.com/10254642/236097701-d6c33915-d891-421d-98d3-6c7843af57e1.png">

Note that Gaussian doesn't generalize well to data points outside of the training set

<img width="562" alt="image" src="https://user-images.githubusercontent.com/10254642/236103414-01aa511c-6f2f-48eb-9377-e11322955adf.png">

But if we apply regularization with a very small alpha, Gaussian works well. The smaller the alpha, the more we penalize complexity

<img width="558" alt="image" src="https://user-images.githubusercontent.com/10254642/236104260-52917e22-70dd-43f4-9057-29e7bedda95c.png">

## Quadratic Regression
Quadratic and Gaussian fit exactly the same.

<img width="588" alt="image" src="https://user-images.githubusercontent.com/10254642/236100212-a72445fa-630f-4600-b9e6-4594d7e212be.png">

Gaussian doesn't generalize

<img width="576" alt="image" src="https://user-images.githubusercontent.com/10254642/236103143-8809920a-958e-4381-8ad5-26c3103beda3.png">

But adding regularization helps

<img width="578" alt="image" src="https://user-images.githubusercontent.com/10254642/236104547-087c3a99-ff4d-4243-af91-53500c498dbc.png">


## Log Regression

Quadratic isn't accurate

<img width="553" alt="image" src="https://user-images.githubusercontent.com/10254642/236100975-0f28af19-28dd-4154-ad4b-2f8f77dd1503.png">

Gaussian doesn't generalize

<img width="568" alt="image" src="https://github.com/atancoder/ml_playground/assets/10254642/3f3233dd-891c-40a4-8176-5c0a3b2c31db">

Guassian w/ regularization

<img width="562" alt="image" src="https://user-images.githubusercontent.com/10254642/236102848-0a40bf34-40fe-45b8-aae1-925c580b79ea.png">

I tried playing around with regularization but I couldn't find parameters that would result in a good fit. The ideal solution would be to transform the data points yourself into log(x) and fit a linear classifier through that.

Project x and y into log()

<img width="604" alt="image" src="https://github.com/atancoder/ml_playground/assets/10254642/daa0d2de-f72b-4bf3-b2f6-1e5353192fb3">

## Sine Regression
Gaussian is the only one that works and fits it perfectly.

<img width="583" alt="image" src="https://user-images.githubusercontent.com/10254642/236101694-05e317e6-4b80-48f2-8756-bfa29cf34867.png">


Gaussian generalizes perfectly this time with no test error. Adding regularization here actually hurts accuracy.

<img width="582" alt="image" src="https://user-images.githubusercontent.com/10254642/236102588-ef191bd8-68fa-423b-a38c-60a0144faf96.png">


## Observations

The Gaussian works well in all datasets, but I noticed it doesn't generalize well to when we see data outside of the range of training examples. For example, look at how it classifies points outside of the range
