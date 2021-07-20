import copy
import inspect
import io
import os
import shutil
import sys
import tempfile
import warnings
import zipfile

import flatbuffers
from tensorflow_lite_support.metadata import metadata_schema_py_generated as _metadata_fb
from tensorflow_lite_support.metadata import schema_py_generated as _schema_fb
from tensorflow_lite_support.metadata.cc.python import _pywrap_metadata_version
from tensorflow_lite_support.metadata.flatbuffers_lib import _pywrap_flatbuffers

model_meta = _metadata_fb.ModelMetadata.GetRootAsModelMetadata(
    metadata_buf, 0)
with _open_file(self._model_file, "rb") as f:
    model_buf = f.read()
model = _schema_fb.Model.GetRootAsModel(model_buf, 0)

num_input_tensors = model.Subgraphs(0).InputsLength()
num_input_meta = model_meta.SubgraphMetadata(0).InputTensorMetadataLength()
if num_input_tensors != num_input_meta:
    raise ValueError(
        "The number of input tensors ({0}) should match the number of "
        "input tensor metadata ({1})".format(num_input_tensors,
                                             num_input_meta))
num_output_tensors = model.Subgraphs(0).OutputsLength()
num_output_meta = model_meta.SubgraphMetadata(
    0).OutputTensorMetadataLength()
if num_output_tensors != num_output_meta:
    raise ValueError(
        "The number of output tensors ({0}) should match the number of "
        "output tensor metadata ({1})".format(num_output_tensors,
                                              num_output_meta))
