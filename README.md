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
https://papers.nips.cc/paper_files/paper/2011/file/218a0aefd1d1a4be65601cc6ddc1520e-Paper.pdf
</p>










