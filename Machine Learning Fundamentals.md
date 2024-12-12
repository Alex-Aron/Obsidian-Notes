# Fundamentals of Deep Learning - Nithin Buduma, Nikhil Buduma, and Joe Papa

## Chapter 1: Fundamentals of Linear Algebra for Deep Learning
### Matrix Operations:
- Already learned from linear algebra but recap:
1. Matrix Multiplication(Not covering addition or subtraction):
	- When multiplying a matrix with another matrix, the number of columns in the first matrix must be equal to the number of rows in the second matrix.
	- Example(more specifically the dot product intepretation):![[Pasted image 20241211121834.png]]
	- Not commutative, but associative
	- Example(column vector interpretation):
		- ![[Pasted image 20241211122011.png]]
		- 
2. Inverse of matrix. Matrix B is an inverse of A if AB = BA = I
3. Matrix vector multiplication (know)

### The Fundamental Spaces:
- Vector Space: The space defined by a list of vectors and all possible linear combinations 
	- Closed under scalar multiplication and addition
- Column Space: Set of all possible vectors v and their products Av. C(A) = column space of A
- Basis of C(A): Check if any vector is linear combination of previous vectors, if so remove it. Provides no extra information. Result is basis vector
- Dimension: Length of the basis
- The basis of any vector space spans the space. Essentially, all elements in the vector space can be formulated as a linear combination of basis vectors. The entirety of R3 can be defined as w1(1,0,0) + w2(0,1,0 ) + w3(0,0, 1). 
- Null space of matrix A N(A): Any vector v such that $A*v=0$ . Other than v=0, any orthogonal vector would also satisfy this property.
	- To find nontrivial solutions, pick a vector $R^n$ not in the row space, then find its projection onto the row space then subtract for null space vector. 

### Eigenvectors and Eigenvalues:
- An eigenvector for a matrix A is a nonzero vector v such that $Av=Cv$ . Roots of $A-cI = 0$ are the eigenvalues of the matrix.   

## Chapter 2: Fundamentals of Probability

### Events and Probability:
- Discrete space -> Finite or countably infinite number of possibilities.
- Sample Space: Entire set of possibilities
- Event: A subset of the sample space(a dice landing on a 2)
- Probability Distribution: Set of probabilities of each event in the sample space, sums to 1.

Frequentist view: After a large number of trials, the probability of an outcome 'emerges'. A dice has a probability of 1/6 on each side because a very large number of rolls will give roughly these proportions
Bayesian View: No prior information from the structure/rolling process that would suggest any side should have a different chance over another. Set of probabilities is termed the *prior* 
- This view allows us to update our prior as more data appears, forming a posterior. 
- This view is often applied to neural networks. Assume each weight has a prior associated, and as training occurs, update the prior associated with each weight to better fit data.

Tenets of probability in the discrete space (all of these are obivous):
- Sum of all event probabilities in the sample space is 1. $∑_o P (o) = 1$.
- $P(E_1)= 1 -P(E_1^c)$. 
- If an event $E_1$ is a subset of the event $E_2$ then $P(E_1) ≤ P(E_2)$. 
- $P(A ∪ B) = P(A) + P(B) − P(A ∩ B)$ 

### Conditional Probability:
- Probability of E given G. $P(E | G)$
- Think of some trained neural nets as conditional probability. With the MNIST database, a network finding the probability an image is 0 is technically $P(0|input)$
- Independence $P(E_1|E_2) = P(E_1)$. 
- Also random variables just map sample space to another space. Example is X(input), where X is the number of heads in a sequence of coin flips. X=3 has the probability P(x=3). 

### Expectation and Variance:
- Expectation of a random variable X can be denoted as E[X]
- $E[X] = ∑_o o^*P (X = o).$
Example usage: In a single coin flip the expected number of heads is $∑_{o∈{0,1}} o^*P (o) = 0*0. 5 + 1*0. 5 = 0. 5$
- Expectation is linear, essentially, given 2 variables A and B:
- ![[Pasted image 20241211131738.png]]
- Which results in $E[A] + E[B]$
- Variance: The average deviation from the expected value in repetitions of an experiment
- Var(X) is equivalent to $E[(X-u)^2]$ where u = E[X]
-  $E[(X-u)^2]$ = $E[X^2] - E[X]^2$
- Variance is cannot be simplified with linearity like expectation. Covariance is present, or the measure of dependence between two random variables. 
### Bayes Theorem:
- $P (B|A) = {P(A|B)P(B)}/P(A)$

### Entropy, Cross-Entropy and KL Divergence:
- Entropy: A metric to encapsulate the uncertainty within a probability distribution.
- Expected number of bits per trial given we have optimized for the distribution p(x)
- $E_p(x) [log_2 1 / p(x) ] = ∑_{xi} p (x_i) log_2 1 / p(x_i) = −∑xi p (x_i) log_2 /p (x_i)$
Dice Example: 
- In a fair dice, the outcome for each can be denoted as 1 as 0, two as '1', three as '10', four as '11', five as '100', and six as '101'. 
- But if the end result is represented as 0110, we have run into an issue. Is it a 1, 2 two's then another 1 or is it 1, 2, 3. Or is it 0 4 0? This is a problem
- A prefix string prevents binary string representation from being prefixes of each other.
- Following the same example, if each side of the dice was 
- Entropy is highest when each outcome probability is equal. This is because if all probabilities are equal, we have no way of knowing if any output will be more likely to appear than another, so it maximizes uncertainty. 
- Cross Entropy: Measurement for the distinctness of two distributions:
-![[Pasted image 20241211133906.png]]
- Expected number of bits per trial given optimized encoding for the incorrect distribution q(x).
KL Divergence: The expected number of extra bits required to represent a trial when using q(x) compared to p(x)
$KL(p||q) = E_{p(x)} [log_2 1/ q(x) − log_2 1 /p(x) ] = Ep(x) [log_2 p(x) / q(x) ]$
- Commonly used in optimizing the difference between learned probability and true probability distributions. For neural network classifiers


### Continuous Probability Distributions
- Probability Density Functions (PDFs)
	- Can almost think of these as integrals
	- Common example, hey I want to get the probability of P(x < 2). But this doesn't work when trying to get the probability of a specific value. If the PDF function is similar to area, we cannot find it at one specific value. This introduces likelihood, which makes a distinction between the likelihood of events and the value given by the PDF function. 
- Gaussian Distribution: $f (x; μ, σ) = \frac{1}{σ√2π} e^ {-1/2 ( x−μ σ )}$
	- So important because highest likelihood also mean outcome
- Bayes Theorem in the Continous Domain![[Pasted image 20241211171814.png]] 
## Chapter 3: The Neural Network
- Linear Perceptron:  A more simple model that makes predictions on a data point x with a parameter vector $\theta$. The model makes predictions with the function H(x, theta). Doesn't really work for a lot of stuff 
- Neuron: Digital representation of the human neuron. Takes inputs x$_1$ ... x$_n$ with corresponding weights w$_1$ ... w$_n$.
- Logit: The output of a neuron $z = ∑n _{i=0} w_ix_i$
- Feed Forward Neural Network: Consists of the input layer with neurons, the output layer with the final values, and any layer in between the two is called a hidden layer. It is not requried for each layer to have the same number of neurons. Inner layers often have less to force the compression of data
	- All inputs and outputs are vectorized representations. An example is an input of vectors where each has 3 values representing an rgb color code. The output could then be a simple vector for the number of values being classified. It could be a [x y] vector where if x is one, it is a cat, and if y is one there is a dog.
	- Feed forward neural networks can be simplified such that all hidden layers can be removed, only giving an input and output layer. This is bad because hidden layers are 'where the magic happens' 

### Nonlinear Neurons
1. Sigmoid Neuron: $f(z)=\frac{1}{1+e^{-z}}$
	1. As z approaches infinity, $\frac{1}{e^{z}}$ becomes essentially 0 and the output is close to one. However, as it approaches 0 the denominator becomes incredibly large and the output is close to 0.
	2. Graph of sigmoid neuron ![[Pasted image 20241211185026.png]]
2. Tahn Neuron: $f(z)=tanh(z)$
	1. Also s shaped like the sigmoid neuron, but it ranges from -1 to 1 instead of 0 to 1
	2. Often preferred over sigmoid because it is centered at 0 instead of .5
3. Restricted Linear Unit(ReLU) Neuron: Uses the function f(z) = max(0, z). This looks like a simple line y=x. Common choice for computer vision
- Softmax Layers: A special kind of output layer highlighting a probability distribution between the outputs. The sum of all values in a softmax neuron should be 1. In a strong, accurate prediction one value will be incredibly close to one, and the rest close to 0. A weaker prediction would be a more equal distribution of probabilities. For example, the softmax output of a vision model determining a handwritten number should have one value very close to one. If a handwritten 9 is the input, the output should give $p_8$ a large value. 

## Chapter 4: Training Feed Forward Neural Networks
- Training: Exposing a neural network to a large number of training examples(data) while modifying the weights to minimize errors.
- The goal of training a neuron is to pick the optimal weights $w_1$ to $w_n$ that create the minimal amount of error. An example of an error that could be minimized is the square error over all training data. Assume that, for the ith training example from the data, t$^{(i)}$ is the true answer and y$^{(i)}$ is the value computed by the neural network, the total error could be represented as:
$E = \frac{1}{2} ∑_i (t ^{(i)} − y ^{(i)}) ^2$
- With this, the model would be 'perfect' if the sum of values is 0, and the closer to zero, the smaller the error, and the better the weights. So our motivation, using the same function h(x, $\theta$), is to make the parameter vector theta a value that brings the error close to 0. 
If the goal is to have a value of 0, why not just create a system of equations and find the RREF(reduce row echelon form)? Well, commonly used neurons are not linear, so a system of equations is not possible.

### Gradient Descent:
- Assume we have a neuron with two inputs. Given the two input weights, and the output of the error function E(defined earlier), we can visualize a shape in 3-d space with points (w1, w2, E).  ![[Pasted image 20241211192147.png]]
	- What it might look like:
- Think of the surface as a set of eliptical contours(each 'slice' at a specific E is an ellipse). The contours correspond to whatever w1 and w2 are set to. The close each ellipse is to eachother, the steeper the slope. The steepest descent happens to be perpendicular to the ellipse contours. This steepest descent is represented as a vector called the gradient. 
```Gradient: The gradient is essentially a vector of partial derivatives that indicates how a multivariate function changes as its input parameters change. In the context of machine learning, it specifically refers to how the loss function (which measures the error or difference between predicted and actual outputs) changes with respect to the model's parameters (like weights and biases)```
- Gradient Descent Algorithm: Thinking of the same 3d space earlier, we can find an algorithm to minimize error. 
	1. Start with a random weight, placing us on the horizontal plane. 
	2. Find the gradient at this point. Take a step in the direction of the steepest descent.
	3. Calculate the new gradient, once again take a step in the direction of the steepest descent. 
	4. Perform until minimum error reached

### Delta Rule and Learning Rates:
On top of neurons, algorithms need parameters as well. These parameters needed for the training process can be called hyperparameters. 
- Learning Rate: 
	- When performing gradient descent, how do we know how much to 'step' in the direction of the gradient? Well, the step distance should depend on how steep it is. The less steep it gets, the closer to the minimum. The learning rate is a factor by which the gradient vector is multiplied to speed up this process. But too large of a rate can cause problems, it can keep jumping past the best option. 
- Delta Rule: How much each weight should be changed. 
- ![[Pasted image 20241211195544.png]]

### Gradient Descent with Sigmoid Neurons:
- The logit z is the sum of the weighted inputs: $z = \sum_kw_kx_k$
- The logit is then put into $f(z)=\frac{1}{1+e^{-z}}$
- $Δw_k = ∑_i ϵx ^{(i)} _k y ^{(i)}(1 − y ^{(i)})(t ^{(i)} − y^{(i)})$
- Very similar to the delta rule, but with extra products to account for the structure of the sigmoid neuron. 

### The Backpropogation Algorithm:
Even if the actions in a hidden layer are unknown, we can measure the change in the gradient based on changes in the hidden activities. But each hidden unit can affect numerous outputs so we have to utilize dynamic programming. Find the error derivates for one layer, then find the error derivates for the next layer, up until the initial input. 
1. Start at the output layer and calculate its error:
	1. ![[Pasted image 20241211201905.png]]
2. Calculate error derivates for layer below it, layer i. The partial derivate of the logit with respect to the incoming output data from the layer beneath is the weight of the connection w$_{ij}$
	1. ![[Pasted image 20241211202600.png]]
3. After this entire routine, the table is filled with all partial derivates(of the error function with respect to hidden unit actions), we can finally determine how the error changed with respect to the weights of each input. this gives us:
	1. ![[Pasted image 20241211202854.png]]
4. Finally, sum all of the parital derivates over every single training example in the dataset:
	1. ![[Pasted image 20241211202925.png]]
5. Backpropogation, more specifcally, this is batch gradient descent.
	1. Only works well for a simple quadratic error surface, it is very sensitive to saddle points and can cause premature convergence. 
Another gradientdescent model is the stochastic gradient descent(SGD), which calculates the error surface based on only one example. So the error surface is dynamic, and it allows us to go through flat regions so we are not stuck at saddle points. 
- But this can take very long if we are only looking at the error incurred one example at a time. 
The solution? Mini-batch gradient descent! Instead of the entire dataset in batch gradient, or one value in stochastic, why not compute the error with a little bit of the data. This is called a minibatch, another hyperparameter that gets the best of both from the two gradient descent methods. The equation would then be:
	![[Pasted image 20241211210410.png]]
	

### Test sets, Validation Sets, and Overfitting:
- Artificial neural networks can get complicated. The MNIST database of 28x28 pixels goes into two hidden layers with 30 neurons, and then a softmax output of 10 neurons. Despite just two hidden layers, the total number of parameters is about 25,000. This can get problematic
- With a given training dataset, it is really easy to fit the model to that data, because we can just give the model enough DoF to where it can just fit to every observation in the training set. But this same model would perform terribly on new data because it was overtuned on the training set. This is called overfitting. 
- Two ways in which overfitting can occur:
	1. Too many neurons. ![[Pasted image 20241211204642.png]]
		1. As there are more and more neurons, the model can be seen overfitting the data. Look at the red area in the 3rd quadrant with 20 neurons. The data encapsulates just two red dots from the training data. In a set of different data, this would not be able to generalize
	2. Too many layers:
		1. ![[Pasted image 20241211204853.png]]
			1. Once again, the more data, the more the model overfits. It is clear the 4 hidden layers would certainly lead to overfitting. 
Three important observation:
- There is always a trade-off between overfitting and model complexity. If it isn't complex enough, it may not be able to solve the problem properly, but if it is too complex it will likely encounter overfitting. 
- It is incredibly misleading to evaluate a model using the training data. Split dataset into training and test data. The test data is essential to ensuring the model can generalize properly.
- After some amount of training, it will stop learning and start overfitting. To prevent this overfitting, we want some way to split up our training process.
	- The epoch is for this. An epoch is a single iteration over the entire training set. If we have training set of size d, with mini-batch gradient descent of batch size b, the epoch would be d/b model updates
	- A validation set will also be used. This is data that tells us how the model performs on data it has not seen after an epoch(this does not change each epoch, it just doesn't impact the model's training so it is 'new' data after each epoch). This validation set will tell us how the model performs. If the training set's accuracy keeps improving but the validation set stays about the same, it is a good sign to stop training before overfitting occurs.

Hyperparameter optimization: We have seen two hyperparameters thus far: learning rate and minibatch size. But how do we get the optimal values for these?
- One way is applying a grid search. Pick a value for each hyperparameter, then train the model with that choice. Do this for all permuations of a certain set of hyperparameter values, then pick the hyperparameters with the best performance on the validation set. 

Workflow for ___building___ and _training_ deep learning models. 
1. Define the problem rigorously. Determine the inputs, any potential outputs, and vectorized representations of both. Picture a model to identify cancer. the input would be an RGB image, which can be a vector of pixel values. The output could be a probability distribution(so a __softmax__ output) of 3 values: normal, benign or malignant. 
2. Then, build a neural network architecture to solve it
	1. The input would be an appropriate size to accept raw data, and the output would only be a size of 3. Then, we would have to define the internal architecture, like hidden layers, connectivities, and more. 
3. Collect data for training and modeling. The data should be properly labeled by a medical expert in this example. 
Workflow for architecture devlopment:
![[Pasted image 20241211211814.png]]

### Preventing Overfitting in Deep Neural Networks:
- Regularization: This modifies the objective function that we minimize by adding addiontal terms that penalize any large weights. Basically, change the objective function so that it becomes Error + $\lambda*f(\theta)$ where f($\theta$) gets larger as theta becomes larger. Lambda is the regularization strength, a new hyperparameter. The value chosen affects how strongly we want to mitigate overfitting. At a value of $\lambda = 0$ then we do not want to take any measures against overfitting. If lambda is too large, then our model will prioritize keeping theta too small instead of finding the values that will truly perform well. 

The common type of regularization is L2 regularization. This involves augmenting the error function with the squared magnitude of all weights in the neural network. For every weight w in the neural network, we add $\frac{1}{2}*\lambda*w^2$ to the error function. 

