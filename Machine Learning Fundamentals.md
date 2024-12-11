# Fundamentals of Deep Learning

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