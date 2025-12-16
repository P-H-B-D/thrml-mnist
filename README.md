# EBM MNIST Training Using THRML
Trains a very simple classifier using an Ising-like (quantized states, floating point interactions & fields / weights & biases) 3 layer Restricted Boltzmann Machine using Equilibrium Propagation. Achieves ~90% accuracy with ~800 DoF / 64 hidden states.  
- `thrml_mnist.ipynb` –> trains MNIST 
- `ffn_baseline.py` & `bnn_ste_baseline.py` –> Feedforward and Straight-Through-Estimator BNN baselines using backprop.
![alt text](https://github.com/P-H-B-D/thrml-mnist/blob/main/eqprop_training_curves.png "Training Curves")