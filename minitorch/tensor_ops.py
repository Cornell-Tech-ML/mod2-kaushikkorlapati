from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import index_to_position, shape_broadcast, to_index, broadcast_index

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function"""
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Map placeholder"""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip placeholder"""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduce Placeholder"""
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`


        Returns:
        -------
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
        ----
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
        -------
            new tensor data

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
        -------
            :class:`TensorData` : new tensor data

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Creates a reduction function that applies a specified binary operation
        across a given dimension of a tensor.

        The resulting function can be called with a tensor and a dimension,
        reducing the tensor along the specified dimension using the provided
        binary operation. The reduction is initialized with a specified start
        value.

        Args:
        ----
            fn (Callable[[float, float], float]): A binary function that takes
                two float arguments and returns a float. This function is used
                to combine elements during the reduction process (e.g., addition,
                multiplication).
            start (float, optional): The initial value to start the reduction.
                Defaults to 0.0. This value is used as the starting point for
                the reduction operation along the specified dimension.

        Returns:
        -------
            Callable[["Tensor", int], "Tensor"]: A function that takes a tensor
            and a dimension to reduce. This function returns a new tensor
            containing the results of the reduction along the specified dimension.

        Example:
        -------
            # Define a simple addition function
            def add(x: float, y: float) -> float:
                return x + y

            # Create a reduction function for addition
            add_reduce = Tensor.reduce(add)

            # Reduce a tensor along dimension 0
            result_tensor = add_reduce(tensor_a, dim=0)

        Notes:
        -----
            The output tensor will have the same shape as the input tensor
            except for the reduced dimension, which will be set to 1.

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of tensor map between
    tensors with possibly different strides.

    Simple version:

    * Fill in the out array by applying fn to each
      value of in_storage assuming out_shape and in_shape
      are the same size.

    Broadcasted version:

    * Fill in the out array by applying fn to each
      value of in_storage assuming out_shape and in_shape
      broadcast. (in_shape must be smaller than out_shape).

    Args:
    ----
        fn: function from float-to-float to apply

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # Iterate over the output tensor's elements.
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        in_index = np.zeros(len(in_shape), dtype=np.int32)

        for i in range(int(operators.prod(out_shape))):
            # Convert i to the correct multidimensional index for the output tensor.
            to_index(i, out_shape, out_index)
            # Use broadcasting rules to get the corresponding input index.
            broadcast_index(out_index, out_shape, in_shape, in_index)
            # Compute the position in the input and output storage arrays.
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)
            # Apply the function to the input value and store it in the output.
            out[out_pos] = fn(in_storage[in_pos])
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError("Need to implement for Task 2.3")

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Args:
    ----
        fn: function mapping two floats to float to apply

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        if (
            np.array_equal(a_shape, b_shape)
            and np.array_equal(a_shape, out_shape)
            and np.array_equal(b_shape, out_shape)
        ):
            for ordinal in range(int(operators.prod(out_shape))):
                out_index = np.zeros(len(out_shape), dtype=np.int32)
                to_index(ordinal, np.array(out_shape, dtype=np.int32), out_index)
                out_pos = index_to_position(
                    np.array(out_index, dtype=np.int32), out_strides
                )
                a_pos = index_to_position(
                    np.array(out_index, dtype=np.int32), a_strides
                )
                b_pos = index_to_position(
                    np.array(out_index, dtype=np.int32), b_strides
                )
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

        else:
            out_index = np.zeros(
                len(out_shape), dtype=np.int32
            )  # Create ndarray with dtype int32
            a_index = np.zeros(
                len(a_shape), dtype=np.int32
            )  # Create ndarray with dtype int32
            b_index = np.zeros(
                len(b_shape), dtype=np.int32
            )  # Create ndarray with dtype int32

            for ordinal in range(int(operators.prod(out_shape))):
                to_index(ordinal, np.array(out_shape, dtype=np.int32), out_index)
                for i in range(len(a_shape)):
                    if a_shape[i] == 1:
                        a_index[i] = 0
                    else:
                        a_index[i] = out_index[i]

                for i in range(len(b_shape)):
                    if b_shape[i] == 1:
                        b_index[i] = 0
                    else:
                        b_index[i] = out_index[i]

                # Convert the multidimensional indices to storage positions
                out_pos = index_to_position(
                    np.array(out_index, dtype=np.int32), out_strides
                )
                a_pos = index_to_position(np.array(a_index, dtype=np.int32), a_strides)
                b_pos = index_to_position(np.array(b_index, dtype=np.int32), b_strides)
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
    ----
        fn: reduction function mapping two floats to float

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        dim_size = a_shape[reduce_dim]
        out_index = np.zeros(len(out_shape), dtype=np.int32)

        for out_pos in range(len(out)):
            res = None
            to_index(out_pos, out_shape, out_index)
            out_storage_pos = index_to_position(out_index, out_strides)

            for reduce_pos in range(dim_size):
                a_index = out_index.copy()
                a_index[reduce_dim] = reduce_pos
                a_storage_pos = index_to_position(a_index, a_strides)

                if res is None:
                    res = a_storage[a_storage_pos]
                else:
                    res = fn(res, a_storage[a_storage_pos])

            out[out_storage_pos] = res

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
