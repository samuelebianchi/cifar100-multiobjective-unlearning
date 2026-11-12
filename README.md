# Pareto HyperNetworks for Multi-Objective Machine Unlearning

This project builds upon the official implementation of [Pareto HyperNetworks](https://github.com/AvivNavon/pareto-hypernetworks).

Original paper: ["Learning the Pareto Front with Hypernetworks"](https://arxiv.org/abs/2010.04104).

<p align="center">
  <img src="https://github.com/AvivNavon/pareto-hypernetworks/blob/master/resources/phn.png" width="350">
</p>

Pareto HyperNetworks (PHN) learn the entire Pareto front using a single model.

This repository extends the original implementation with additional code and experiments focused on **multi-objective machine unlearning on CIFAR-100**.

## Install

Clone this repository and install its dependencies:

```bash
git clone https://github.com/samuelebianchi/cifar100-multiobjective-unlearning.git
cd cifar100-multiobjective-unlearning
pip install -e .
```

## Run Experiments

To run the experiments, follow the instructions provided in the `README.md` files within each experiment folder.

The additional unlearning experiments focus on multi-objective machine unlearning on CIFAR-100 using Pareto HyperNetworks.

## Citation

If you find the original Pareto HyperNetworks implementation useful in your research, please consider citing the following paper:

```bibtex
@inproceedings{
navon2021learning,
title={Learning the Pareto Front with Hypernetworks},
author={Aviv Navon and Aviv Shamsian and Gal Chechik and Ethan Fetaya},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=NjF772F4ZZR}
}
```

## Acknowledgements

This project is based on the official implementation of **Pareto HyperNetworks**, originally developed by Aviv Navon, Aviv Shamsian, Gal Chechik, and Ethan Fetaya.

- Original repository: https://github.com/AvivNavon/pareto-hypernetworks
- Original paper: *Learning the Pareto Front with Hypernetworks* (ICLR 2021).

The original implementation is distributed under the MIT License. The original copyright and license notices are retained in this repository. Please refer to the `LICENSE` file for the full license text.

The additional code and experiments developed in this repository extend the original implementation to investigate multi-objective machine unlearning on CIFAR-100.