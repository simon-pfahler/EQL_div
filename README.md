# EQL&div; implementation in TensorFlow2

This project implements the EQL&div; network for neural symbolic regression.
The basis for this implementation are the paper by [Martius and Lampert](http://arxiv.org/abs/1610.02995) introducing the EQL network, and the subsequent work by [Sahoo et al.](https://proceedings.mlr.press/v80/sahoo18a.html) introducing the EQL&div; network.

The script `EQL_div.py` implements the custom architecture of the EQL&div; network. `formula_writer.py` provides a function to write a learned network down as a symbolic formula. In `model_selection.py`, the method for selecting the optimal model among possible candidates is implemented. `train_on_function.py` provides a simple interface for tests, where a target function is given, which is learned using a given network structure.
