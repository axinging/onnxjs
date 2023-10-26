import warnings
from typing import Any, Dict, NamedTuple, Union, cast

import numpy as np

from onnx import OptionalProto, SequenceProto, TensorProto


class TensorDtypeMap(NamedTuple):
    np_dtype: str
    storage_dtype: int
    name: str


# tensor_dtype: (numpy type, storage type, string name)
TENSOR_TYPE_MAP = {
    int(TensorProto.FLOAT): TensorDtypeMap(
        ("float32"), int(TensorProto.FLOAT), "TensorProto.FLOAT"
    ),
    int(TensorProto.UINT8): TensorDtypeMap(
        ("uint8"), int(TensorProto.INT32), "TensorProto.UINT8"
    ),
    int(TensorProto.INT8): TensorDtypeMap(
        ("int8"), int(TensorProto.INT32), "TensorProto.INT8"
    ),
    int(TensorProto.UINT16): TensorDtypeMap(
        ("uint16"), int(TensorProto.INT32), "TensorProto.UINT16"
    ),
    int(TensorProto.INT16): TensorDtypeMap(
        ("int16"), int(TensorProto.INT32), "TensorProto.INT16"
    ),
    int(TensorProto.INT32): TensorDtypeMap(
        ("int32"), int(TensorProto.INT32), "TensorProto.INT32"
    ),
    int(TensorProto.INT64): TensorDtypeMap(
        ("int64"), int(TensorProto.INT64), "TensorProto.INT64"
    ),
    int(TensorProto.BOOL): TensorDtypeMap(
        ("bool"), int(TensorProto.INT32), "TensorProto.BOOL"
    ),
    int(TensorProto.FLOAT16): TensorDtypeMap(
        ("float16"), int(TensorProto.UINT16), "TensorProto.FLOAT16"
    ),
    # Native numpy does not support bfloat16 so now use float32.
    int(TensorProto.BFLOAT16): TensorDtypeMap(
        ("float32"), int(TensorProto.UINT16), "TensorProto.BFLOAT16"
    ),
    int(TensorProto.DOUBLE): TensorDtypeMap(
        ("float64"), int(TensorProto.DOUBLE), "TensorProto.DOUBLE"
    ),
    int(TensorProto.COMPLEX64): TensorDtypeMap(
        ("complex64"), int(TensorProto.FLOAT), "TensorProto.COMPLEX64"
    ),
    int(TensorProto.COMPLEX128): TensorDtypeMap(
        ("complex128"), int(TensorProto.DOUBLE), "TensorProto.COMPLEX128"
    ),
    int(TensorProto.UINT32): TensorDtypeMap(
        ("uint32"), int(TensorProto.UINT32), "TensorProto.UINT32"
    ),
    int(TensorProto.UINT64): TensorDtypeMap(
        ("uint64"), int(TensorProto.UINT64), "TensorProto.UINT64"
    ),
    int(TensorProto.STRING): TensorDtypeMap(
        ("object"), int(TensorProto.STRING), "TensorProto.STRING"
    ),
    # Native numpy does not support float8 types, so now use float32 for these types.
    int(TensorProto.FLOAT8E4M3FN): TensorDtypeMap(
        ("float32"), int(TensorProto.UINT8), "TensorProto.FLOAT8E4M3FN"
    ),
    int(TensorProto.FLOAT8E4M3FNUZ): TensorDtypeMap(
        ("float32"), int(TensorProto.UINT8), "TensorProto.FLOAT8E4M3FNUZ"
    ),
    int(TensorProto.FLOAT8E5M2): TensorDtypeMap(
        ("float32"), int(TensorProto.UINT8), "TensorProto.FLOAT8E5M2"
    ),
    int(TensorProto.FLOAT8E5M2FNUZ): TensorDtypeMap(
        ("float32"), int(TensorProto.UINT8), "TensorProto.FLOAT8E5M2FNUZ"
    ),
}

def tensor_dtype_to_json_dtype(tensor_dtype: int) -> str:
    """
    Convert a TensorProto's data_type to corresponding data_type for storage.

    :param tensor_dtype: TensorProto's data_type
    :return: data_type for storage
    """
    return TENSOR_TYPE_MAP[tensor_dtype].np_dtype
