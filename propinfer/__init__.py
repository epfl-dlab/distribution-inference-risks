"""propinfer is a modular framework to run Property Inference Attacks on Machine Learning models.

To run an experiment, you simply should define subclasses of `Generator` and `Model`
to represent your data source and your evaluated model respectively.

Logging is available for this framework, using logger `propinfer`.
"""

import logging

from propinfer.experiment import Experiment
from propinfer.generator import Generator, GaussianGenerator, IndependentPropertyGenerator, ProbitGenerator, \
                                LinearGenerator, SubsamplingGenerator, MultilabelProbitGenerator
from propinfer.model import Model, LinReg, LogReg, MLP

logging.getLogger('propinfer').addHandler(logging.NullHandler())

__pdoc__ = {
    'deepsets': False,
    'model_utils': False
}
