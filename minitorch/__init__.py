"""Provides as the main entry point for various functionalities related
to tensor operations, automatic differentiation, optimization, and testing in
a machine learning framework.

The following components are imported:

- **Testing Utilities**: Classes for testing mathematical operations and
  variables.

- **Tensor Data Structures**: Core tensor implementations and associated
  operations, including tensor data, tensor operations, and tensor
  functions.

- **Datasets**: Utilities for handling and providing datasets for model
  training and evaluation.

- **Optimization Algorithms**: Implementations of various optimization
  algorithms used for training models.

- **Neural Network Modules**: Definitions of neural network layers and
  models.

- **Automatic Differentiation**: Utilities for computing gradients
  automatically.

- **Scalar Operations**: Functions and operations specifically for scalar
  values.

This module encapsulates the key functionalities needed for building and
training machine learning models, providing essential tools for tensor
manipulation, gradient calculation, and model optimization.
"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
