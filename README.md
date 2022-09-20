# Distribution Inference Risks: Identifying and Mitigating Sources of Leakage

In this repository, we present the code used to run the different experiments in the paper.

To install the environment, you can use conda with the command:

```conda env create -f environment.yml```

Please note that PyTorch is required to run this framework. Please find installation instructions corresponding to you [here](https://pytorch.org/).

## Library

For these experiments, we use a property inference library we developed in the scope of our research. It is available on [PyPi](https://pypi.org/project/propinfer/) and [GitHub](https://github.com/epfl-dlab/property-inference-attacks).

## Citation

The research paper relative to this library is currently only available as a pre-print on [arXiv](https://arxiv.org/abs/2209.08541). If you use our library for your research, please cite our work:

```
V. Hartmann, L. Meynent, M. Peyrard, D. Dimitriadis, S. Tople, and R. West, Distribution inference risks: Identifying and mitigating sources of leakage. arXiv, 2022. doi: 10.48550/ARXIV.2209.08541. 
```

```
@misc{https://doi.org/10.48550/arxiv.2209.08541,
	title        = {Distribution inference risks: Identifying and mitigating sources of leakage},
	author       = {Hartmann, Valentin and Meynent, Léo and Peyrard, Maxime and Dimitriadis, Dimitrios and Tople, Shruti and West, Robert},
	year         = 2022,
	publisher    = {arXiv},
	doi          = {10.48550/ARXIV.2209.08541},
	url          = {https://arxiv.org/abs/2209.08541},
	copyright    = {arXiv.org perpetual, non-exclusive license},
	keywords     = {Cryptography and Security (cs.CR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences}
}
```
