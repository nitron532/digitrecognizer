<h1>Multithreaded MNIST Handwritten Digit Neural Network </h1> 
<h2>Implemented using C++, Eigen library for linear algebra, some extra utilities for loading images</h2>
<h2> Loss function visualizer using NumPy and Matplotlib </h2>
<p>
Loss: Categorical Cross Entropy <br>
Activations: ReLu (hidden layers), Softmax (output layer) <br>
Initialization: He <br>
L2 Regularization <br>
Dropout <br>
Highest accuracy achieved 98%
</p>
<p>implements Hogwild! parallel SGD (lock free parallelism), despite concurrent access of shared memory, accuracy isn't largely affected <br>

Using amount of cores on your machine might not be the best choice:
  1. If a large amount, OS overhead
  2. False sharing of cache line may cause blocking of threads (weight and bias Eigen matrices are contiguous in memory)
</p>

<li>
  <ul>looking into elemental for parallel matrix mult</ul>
  <ul>delete irrelelvant loss values from loss plot to speed up updating</ul>

</li>














