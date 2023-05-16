# Distribution Inference Risks: Identifying and Mitigating Sources of Leakage

In this repository, we present the code used to run the different experiments in the paper.

To install the environment, you can use conda with the command:

```conda env create -f environment.yml```

Please note that PyTorch is required to run this framework. Please find installation instructions corresponding to you [here](https://pytorch.org/).

## Library

For these experiments, we use a property inference library we developed in the scope of our research. It is available on [PyPi](https://pypi.org/project/propinfer/) and [GitHub](https://github.com/epfl-dlab/property-inference-attacks).

## Citation

If you use this code for your work, please cite our paper as follows:

```
V. Hartmann, L. Meynent, M. Peyrard, D. Dimitriadis, S. Tople, and R. West, Distribution inference risks: Identifying and mitigating sources of leakage. IEEE Conference on Secure and Trustworthy Machine Learning, 2023.
```

```
@article{hartmann2023distribution,
	title        = {Distribution inference risks: Identifying and mitigating sources of leakage},
	author       = {Hartmann, Valentin and Meynent, Léo and Peyrard, Maxime and Dimitriadis, Dimitrios and Tople, Shruti and West, Robert},
	year         = {2023},
	journal      = {IEEE Conference on Secure and Trustworthy Machine Learning}
}
```
