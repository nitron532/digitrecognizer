<h1>Multithreaded MNIST Handwritten Digit Neural Network </h1> 
<h2>Implemented using C++, Eigen library for linear algebra, some extra utilities for loading images</h2>
<h2> Loss function visualizer using NumPy and Matplotlib </h2>
<p>
Loss: Categorical Cross Entropy <br>
Activations: ReLu (hidden layers), Softmax (output layer) <br>
Initialization: He <br>
L2 Regularization <br>
Highest accuracy achieved 98%
</p>
<p>implements Hogwild! parallel SGD (lock free parallelism) <br> 
  Disabled OpenMP multithreading for matrix ops (overhead for smaller / medium size matrix operations > speedup)
  Single threaded training is actually faster than multithreaded. Maybe because:
  False sharing during multithreaded training FeedForward weight/bias access, mostlikely many attempted parallel accesses on the same cache line
  Also Eigen is super optimized (taking into account compiler optimization flags)
</p>

<li>
  delete irrelelvant loss values from loss plot to speed up updating
</li>

















