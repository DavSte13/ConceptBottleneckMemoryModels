# ConceptBottleneckMemoryModels

This repository contains code for the Paper "Learning to Intervene on Concept Bottlenecks". It provides an implementation of concept bottleneck memory models (CB2M) based on concept bottleneck models (CBM).
The repository furthermore contains code to evaluate CB2M on several different datasets: CUB, Parity MNIST, Parity Color MNIST and Parity SVHN. 

## Installation:

* Install the `requirements.txt`
* The experiments on CUB are based on the paper "Concept Bottleneck Models" (Koh et al. 2020). To perform experiments on CUB, download the official CUB dataset (CUB_200_2011), pretrained inception v3 network (pretrained) and the processed CUB data (CUB_processed) from their codalab sheet (https://worksheets.codalab.org/worksheets/0x362911581fcd4e048ddfd84f47203fd2)

## Usage:

To perform the experiments desribed in the paper, the script `experiments.sh` is available. This exemplary contains commands to run all the experiments (for the fold 0). The general pipeline used for the experiments is the following:

* Training of the base CBM
* Precomputing data for the CB2M experiments
* Hyperparameter optimization
* Detection (and Performance): Evaluate the detection of mistakes and the performance of interventions after these mistakes
* Generalization: Evaluate the generalization of interventions to new data points.

Further details about run parameters can be found in `experiments.sh` and in the respective files.
