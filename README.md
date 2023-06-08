# Distribution Inference Risks: Identifying and Mitigating Sources of Leakage

In this repository, we present the code used to run the different experiments in the paper.

To install the environment, you can use conda with the command:

```conda env create -f environment.yml```

Please note that PyTorch is required to run this framework. Please find installation instructions corresponding to you [here](https://pytorch.org/).

## Library

For these experiments, we use a property inference library we developed in the scope of our research. It is available on [PyPi](https://pypi.org/project/propinfer/) and [GitHub](https://github.com/epfl-dlab/property-inference-attacks).

## Citation

If you use this library for your work, please cite [our paper](https://doi.org/10.1109/SaTML54575.2023.00018) as follows:

```
V. Hartmann, L. Meynent, M. Peyrard, D. Dimitriadis, S. Tople and R. West, "Distribution Inference Risks: Identifying and Mitigating Sources of Leakage," 2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), Raleigh, NC, USA, 2023, pp. 136-149, doi: 10.1109/SaTML54575.2023.00018.
```

```
@INPROCEEDINGS{10136150,
  author={Hartmann, Valentin and Meynent, Léo and Peyrard, Maxime and Dimitriadis, Dimitrios and Tople, Shruti and West, Robert},
  booktitle={2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)}, 
  title={Distribution Inference Risks: Identifying and Mitigating Sources of Leakage}, 
  year={2023},
  volume={},
  number={},
  pages={136-149},
  doi={10.1109/SaTML54575.2023.00018}
}
```
