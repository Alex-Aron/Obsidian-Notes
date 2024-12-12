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
This encourages the network to use all of the inputs a little rather than only using some a lot. It is often also called weight decay because the gradient descent goes linearly to 0.
![[Pasted image 20241212130912.png]]
You can see how lamda values can solve overfitting. In the first $\lambda$ is small enough that the model evidently overfits. 

Another type is L1 regularization, where $\lambda*|w|$ is added for each weight. This typically leads to a neuron ignoring many inputs, and only focusing on a small subset of the inputs. This can be useful if you want to see what inputs contribute to a decision. L2 almost always better

-- Max norm constraints -- Same goal of reducing theta so it does not become too large. These force an upper bound on the weight of every neuron, using projected gradient to enforce this constraint. After a gradient descent step is performed, the new weight is compared to a radius c. If it is greater than c, the vector is projected onto the ball with the radius c. 

Dropout: A neuron is active with some probability p, and if it is zero it is inactive. This forces the network to be accurate in the absence of certain information by preventing it from becoming too dependent on any certain neuron. 
![[Pasted image 20241212132959.png]]


## Chapter 5: Implementing Neural Networks in PyTorch

### PyTorch Tensors:
- Tensors are the primary structure where numerical information is stored/manipulated. 
	- Tensors can be thought of as a generalization of 2d matrices and 1d arrays and are capable of storing multidimensional data(example: batchers of 3 channel images)
		- The data storage in the example would be 4 dimensions, so an index could be present for each individual image
- Capable of using dimensions past the 4d space, but uncommon.
- Tensors are universal. They represent the input to models, the weight layers within the models, and the output of the models. Any standard operation like addition, multiplication, inversion and more can be run on tensors.

#### Tensors Initialization:
1. The first way to initialize tensors is with simple lists or numerical primitives:
```python
arr = [1,2]
tensor = torch.tensor(arr)
val = 2.0
tensor = torch.tensor(val)```
2. They can also be initialized from numpy arrays:
```python
import numpy as np
np_arr = np.array([1,2])
x_t = torch.from_numpy(np_arr)
```
3. Finally, through PyTorch functions themselves
```python
zeros_t = torch.zeros((2,3))
ones_t = torch.ones((2,3))
rand_t = torch.randn((2,3))
```

#### Tensor Attributes:
- The dimensionality in init 3 is the same as the number of indices in the tupple.
	- The shape attribute allows us to get the dimensionality of a tensor: tensor.shape
- Data type being stored. Floats, complex #'s, integers, booleans
	- torch.dtype
- Device attribute: tensor.device gives the device it is being run on (type='cpu') by default
	- If GPU available use CUDA so much faster
- The .to function allows us to modify an attribute inthe tensor, like data type.

#### Tensor Operations:
1. Multiplying a tensor by a scalar 
```python
c = 10
x_t = x_t*c #underscore t will be a tensor from here on out
```
2. Adding/Subtracting two tensors:
```python
x1_t = torch.zeros((1,2))
x2_t = torch.ones((1,2))
x1_t + x2_t 
```
These still adhere to the same rules for matrix addition, the tenors are the same dimension. If they are not, and the inputs are broadcastable, they will be broadcasted. Check pytorch docs for more info
3. Tensor Multiplication. For dimensionalities less than or equal to 2, it is identical to matrix and vector multiplication. But it also works on higher dimensions as well. Imagine two tensors, one of shape (2,1,2) and another of shape (2,2,2). The first tensor can be thought of as a list of length two, where each value in the list is a 1x2 matrix. The second is a list of length two where each value in the list is a 2x2 matrix. The product of these two tensors is a length two list where index i is the product of first tensor index i and the second tensor index i. 
	![[Pasted image 20241212143205.png]] 
	Visualization of the example described. 
	This shows how 3d tensors can be multiplied
Generalizing this to four dimensions, it can be thought of similarly. Instead of a list of matrices, the 4d tensor is like a 2d array (a grid) of matrices. The (i,j)-th index is the matrix product of the same index in the two 4d input tensors that are being multiplied
This can be represented as $P_{i,j,x,z} = \Sigma _yA_{i,j,x,y}*B_{i,j,x,y}$
To perform this multiplication, use the matmul function. Also, note that the procedure for the 3d/4d arrays can work in any dimensionality as long as the two input tensors follow the constraints. Broadcasting also works here. 
Broadcasting: Essentially, broadcasting is the process by which tensors can be automatically expanded to equal sizes in order to perform operations. 
```python
# matmul example
x1_t = torch.tensor([1,2],[3,4])
x2_t = torch.tensor([1,2,3],[4,5,6])
torch.matmul(x1_t, x2_t)
```
4. Tensor indexing is similar to any other list, also similar to numpy
```python
i,j,k = 0,1,1
x3_t = torch.tensor([[[3,7,9],[2,4,5]],[[8,6,2],[3,9,1]]])
print(x3_t)
# out:
	# tensor([[[3, 7, 9],
	# [2, 4, 5]],
	# [[8, 6, 2],
	# [3, 9, 1]]])
x3_t[i,j,k]
# out:
	# tensor(4)

# To get a larger portion: (the two lines below are logically equivalent)
x3_t[0] # Returns the matrix at position 0 in tensor
x3_t[0,:,:] # Also returns the matrix at position 0 in tensor!
# ':' usage is similar to standard python.
# out:
	# tensor([[3, 7, 9],
	# [2, 4, 1]])
```

### Gradients in PyTorch:
- Recall partial derivatives. 
If a function took in three inputs, like $f(x,y,z)= x^2+y^2+z^2$ then the gradient would be 
$g = [2x 2y 2z]$ 
In pytorch this might look like:
```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(1.5, requires_grad=True)
f = x**2 + y**2 + z**2
f.backward()
x.grad, y.grad, z.grad
#would print (tensor(4.), tensor(6.), tensor(3.))
```
In a neural network, instead of getting the partial derivative of x for the gradient, we instead represent the neural network as f(x,$\theta$) . Then we compute the gradient of the loss of f with respect to theta. Then, adjustments can be made based on the gradient until a proper theta is found.  

### The PyTorch nn Module:
- Simply import with 'import torch.nn as nn'
 A simple initialization for a feed forward neural network might look like:
```python
in_dim, out_dim = 256, 10
vec = torch.randn(256)
layer = nn.Linear(in_dim, out_dim, bias=True)
out = layer(vec)
```
The code creates a single layer with bias in a feedforward neural network. It takes in a vector with dimension 256 and outputs a vector with dimension 10.
A feedforward neural network in pytorch is just a composition of layers. 
example:
```python
in_dim, feature_dim, out_dim = 784, 256, 10
vec = torch.randn(784)
layer1 = nn.Linear(in_dim, feature_dim, bias=True)
layer2 = nn.Linear(feature_dim, out_dim, bias=True)
out = layer2(layer1(vec))

#nonlinearity:
relu = nn.ReLU()
out = layer2(relu(layer1(vec)))

```
However this is still linear, and as discussed earlier in the notes, we want nonlinearity to be supported. 
The nn module also have ReLU and tanh. 

nn.module class is the basis for all neural networks in pytorch.
Building on the exapmle earlier:
```python
class BaseClassifier(nn.Module):
	def __init__(self, in_dim, feature_dim, out_dim):
		super(BaseClassifier,self).__init__()
		self.layer1 = nn.Linear(in_dim, feature_dim, bias=True)
		self.layer2 = nn.Linear(feature_dim, out_dim, bias=True)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.layer1(x)
		x = self.relu(x)
		out = self.layer2(x)
		return out
```
Use of this model might look like:
```python
no_examples = 10
in_dim, feature_dim, out_dim = 784, 256, 10
x = torch.randn((no_examples,in_dim))
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
out = classifier(x) ## this calls the forward function
```
Training a module requires a loss metric. This can be done in pytorch with nn as well:
```python
loss = nn.CrossEntropyLoss()
target = torch.tensor([0,3,2,8,2,9,3,7,1,6])
computed_loss = loss(out, target)
computed_loss.backward()
```
The torch.optim module gives us an optimizer module to determine the best optimizer. It updates the parameters for us as well
```python
from torch import optim
lr = 1e-3
optimizer = optim.SGD(classifier.parameters(), lr=lr)
##creates optimizer that can update parameters
optimizer.step() # Perform the SGD and update the parameters 
optmizer.zero_grad() # 0 out gradients between minibatches
## Updates the parameters of classifier through SGD
```

### PyTorch Datasets and Dataloaders:
- The pytorch dataset is a class that allows us to access our data. 
- An example dataset for the MNIST dataset of handwritten numbers is:
```python
import os
from PIL import Image
from torchvision import transforms

class ImageDataset(Dataset):
	def __init__(self, img_dir, label_file):
		super(ImageDataset, self).__init__()
		self.img_dir = img_dir 
		self.labels = torch.tensor(np.load(label_file, allow_pickle=True))
		self.transforms = transforms.ToTensor()

	def __get__item(self, idx):
		img_pth = os.path.join(self.img_dir, "img_{}.jpg".format(idx))
		img = Image.open(img_pth)
		label = self.labels[idx]
		return {"data": img, "label": label}

	def __len__(self):
		return len(self.labels)
```
This gets images from a dataset following the naming convention img-idx.png with idx being the index. Also assumes ground-truth labels are in a numpy array that can be indexed with idx to find image labels. 

- The dataloader module takes in a dataset instance, like the one above, and automatically loads the dataset into the minibatch and shuffles the dataset between epochs. 
```python
train_dataset = ImageDataset(img_dir='./data/train/', label_file='./data/train/labels.npy')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

for minibatch in train_loader:
	data, labels = minibatch['data'], minibatch['label']
	out = classifier(data)
	print(data)
	print(labels)

```


### MNIST Classifier in PyTorch:
```python
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

class BaseClassifier(nn.Module):
 def __init__(self, in_dim, feature_dim, out_dim):
 super(BaseClassifier, self).__init__()
 self.classifier = nn.Sequential(
 nn.Linear(in_dim, feature_dim, bias=True),
 nn.ReLU(),
 nn.Linear(feature_dim, out_dim, bias=True)
 )

 def forward(self, x):
 return self.classifier(x)
# Load in MNIST dataset from PyTorch
train_dataset = MNIST(".", train=True,
 download=True, transform=ToTensor())
test_dataset = MNIST(".", train=False,
 download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset,
 batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset,
 batch_size=64, shuffle=False)

# Instantiate model, optimizer, and hyperparameter(s)
in_dim, feature_dim, out_dim = 784, 256, 10
lr=1e-3
loss_fn = nn.CrossEntropyLoss()
epochs=40
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
optimizer = optim.SGD(classifier.parameters(), lr=lr)
def train(classifier=classifier,
 optimizer=optimizer,
 epochs=epochs,
 loss_fn=loss_fn):
 classifier.train()
 loss_lt = []
 for epoch in range(epochs):
	 running_loss = 0.0
 for minibatch in train_loader:
	 data, target = minibatch
	 data = data.flatten(start_dim=1)
	 out = classifier(data)
	 computed_loss = loss_fn(out, target)
	 computed_loss.backward()
	 optimizer.step()
	 optimizer.zero_grad()
	 # Keep track of sum of loss of each minibatch
	 running_loss += computed_loss.item()
 loss_lt.append(running_loss/len(train_loader))
 print("Epoch: {} train loss: {}".format(epoch+1,
 running_loss/len(train_loader)))
 plt.plot([i for i in range(1,epochs+1)], loss_lt)
 plt.xlabel("Epoch")
 plt.ylabel("Training Loss")
 plt.title(
 "MNIST Training Loss: optimizer {}, lr {}".format("SGD",
lr))
 plt.show()
 # Save state to file as checkpoint
 torch.save(classifier.state_dict(), 'mnist.pt')
def test(classifier=classifier,
	 loss_fn = loss_fn):
	 classifier.eval()
	 accuracy = 0.0
	 computed_loss = 0.0
 with torch.no_grad():
	 for data, target in test_loader:
		 data = data.flatten(start_dim=1)
		 out = classifier(data)
		 _, preds = out.max(dim=1)
		 # Get loss and accuracy
		 computed_loss += loss_fn(out, target)
		 accuracy += torch.sum(preds==target)

	 print("Test loss: {}, test accuracy: {}".format(
		 computed_loss.item()/(len(test_loader)*64),
		 accuracy*100.0/(len(test_loader)*64)))
```

