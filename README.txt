## QUCONE

The aim of this thesis was to develop and compare the capabilities of a hybrid convolutional neural network (CNN) against those of a traditional CNN. Initially, significant attention was given to optimizing the traditional network to mitigate overfitting. This optimization process had to consider the constraint of the training dataset being limited to only 800 images, due to available computational resources.

The first model analyzed was the Real Amplitudes + ZZ Feature Map (RA + ZZ), already known in literature and used in binary classification. Subsequently, five hybrid models were developed, following an experimental approach as there is no universal rule to establish a priori which model is the best.

The accuracy results of the RA + ZZ model were surpassed by all five subsequent hybrid models. The five hybrid models proposed in this thesis provided competitive results with those of the traditional model on final accuracies (validation, test, training). In particular, one of the models proved to be more effective than the traditional model in terms of test dataset accuracy, although this result is inconclusive due to statistical fluctuations stemming from the limited dataset size.

From the graphs, a common characteristic emerged among all six models, namely the presence of overfitting after a few epochs. Regarding learning capabilities on the training set, the hybrid models outperformed the traditional model for a low number of epochs. In particular, three out of the six models showed significantly better accuracy than the traditional model for a number of epochs around 20.

It is important to emphasize that the study of accuracies should also be extended to the validation set, in order to establish if there is an actual quantum advantage for a low number of epochs.

The results of the hybrid models were obtained using the statevector simulator of the qiskit platform. It is assumed that these results faithfully reflect those obtainable using a modern quantum processor. This is because the quantum circuits developed and used for the simulation are of small dimensions, comprising a reduced number of qubits (two or three) and a limited number of logic gates. These circuits are therefore suitable for "noisy-intermediate scale" devices, where a certain percentage of error occurs in the operations performed, generally on the order of 1% or lower.

Finally, below are listed some aspects through which the study could be further deepened and improved:

- Increase the size of the training dataset: This would entail greater generalization capability and, consequently, a reduction in overfitting for all models.
- Increase the number of epochs until convergence to a fixed value of the learning curve. In this thesis, this was not possible due to the limited dataset, leading to the use of early stopping.
- Further optimize the neural networks: It would be interesting to understand how the results vary with different parameters such as the optimizer, learning rate, and loss function.
- Considering that the dataset has been preprocessed and therefore some information has necessarily been lost (section 3.2), it would be interesting to understand how this influences accuracy results.

## Model Repository

This repository contains various trained models used in the project. For additional models, please refer to the accompanying thesis.pdf file.

### Model Overview

In the "models" folder, you'll find three relevant models that have been uploaded. Each model is accompanied by a "ModelResult" folder, which contains essential evaluation data and graphs.

#### ModelResult Folder Contents:

- **AT**: Accuracy Training (Accuracy on the training set).
- **LVT**: Loss Validation/Training (Loss curves on the training and validation sets).
- **ALV**: Accuracy/Loss Training (Accuracy and loss curves for the training set).

Additionally, each ModelResult folder contains:

- A `.txt` file containing all the data used to produce the graphs.
- A `.pt` file containing the trained PyTorch model.

### Usage

To load a trained PyTorch model from the ".pt" file, use the following command:

```python
hybridModel = torch.load(ModelFolder + hybridModelName + "name.pt")
