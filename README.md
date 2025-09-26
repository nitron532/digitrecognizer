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

Literature has shown that Hogwild! is slower than a single threaded SGD implementation when using a large amount of threads ( > 10) <br>
<img width="550" height="449" alt="image" src="https://github.com/user-attachments/assets/d2be6b45-5390-4f7e-a394-07059569b496" /> <br>
image from: HogWild++: A New Mechanism for Decentralized Asynchronous Stochastic Gradient Descent by Huan Zhang et al <br>
https://ieeexplore.ieee.org/document/7837887 <br>
Be aware of this limitation, due to the implementation of this network using one centralized matrix of weights (Hogwild!) rather than a decentralized, asynchronous updating of weights (Hogwild!++)<br>
This implementation also automatically spawns an amount of threads corresponding to the amount of cores detected by std::thread::hardware_concurrency(). Consult graph for how this might effect performance on your machine.
</p>












