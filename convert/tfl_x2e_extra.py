# SPDX-License-Identifier: Apache-2.0
"""Extra TFLite -> TF handlers for tf2onnx (installed into tf2onnx/tflite_handlers/ by patch_tf2onnx.py).

tf2onnx 1.17.0 has no handler for three ops that LiteRT (litert-torch) emits, so it leaves them as TFL_* nodes
and the ONNX fails downstream ("No Op registered for TFL_GELU with domain_version of 17", seen on 5 zoo models
at Qualcomm AI Hub, 2026-10-30):
  TFL_GELU         -> 0.5 * x * (1 + erf(x / sqrt 2)), or the tanh form when `approximate` is set
  TFL_RELU_0_TO_1  -> ClipByValue(x, 0, 1)
  TFL_RELU_N1_TO_1 -> ClipByValue(x, -1, 1)
"""

import numpy as np

from tf2onnx import utils
from tf2onnx.handler import tfl_op

# pylint: disable=unused-argument,missing-docstring,invalid-name


def _clip(ctx, node, lo, hi):
    np_dtype = utils.map_onnx_to_numpy_type(ctx.get_dtype(node.input[0]))
    lo_c = ctx.make_const(utils.make_name("clip_lo"), np.array(lo, np_dtype)).output[0]
    hi_c = ctx.make_const(utils.make_name("clip_hi"), np.array(hi, np_dtype)).output[0]
    ctx.replace_inputs(node, [node.input[0], lo_c, hi_c])


@tfl_op(["TFL_RELU_0_TO_1"], tf_op="ClipByValue")
class TflRelu0To1:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        _clip(ctx, node, 0, 1)


@tfl_op(["TFL_RELU_N1_TO_1"], tf_op="ClipByValue")
class TflReluN1To1:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        _clip(ctx, node, -1, 1)


@tfl_op(["TFL_GELU"], tf_op="Mul")
class TflGelu:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        x = node.input[0]
        dtype = ctx.get_dtype(x)
        shape = ctx.get_shape(x)
        np_dtype = utils.map_onnx_to_numpy_type(dtype)
        approximate = bool(node.get_attr_value("approximate", 0))
        if "approximate" in node.attr:
            del node.attr["approximate"]

        def const(v, name):
            return ctx.make_const(utils.make_name(name), np.array(v, np_dtype)).output[0]

        def ew(op, inputs, name):
            return ctx.make_node(op, inputs, name=utils.make_name(name), shapes=[shape], dtypes=[dtype]).output[0]

        half_x = ew("Mul", [x, const(0.5, "gelu_half")], "gelu_half_x")
        if approximate:
            x2 = ew("Mul", [x, x], "gelu_x2")
            x3 = ew("Mul", [x2, x], "gelu_x3")
            inner = ew("Add", [x, ew("Mul", [x3, const(0.044715, "gelu_k")], "gelu_kx3")], "gelu_inner")
            t = ew("Tanh", [ew("Mul", [inner, const(np.sqrt(2.0 / np.pi), "gelu_c")], "gelu_scaled")], "gelu_tanh")
            gate = ew("Add", [t, const(1.0, "gelu_one")], "gelu_gate")
        else:
            e = ew("Erf", [ew("Mul", [x, const(1.0 / np.sqrt(2.0), "gelu_inv_sqrt2")], "gelu_xs")], "gelu_erf")
            gate = ew("Add", [e, const(1.0, "gelu_one")], "gelu_gate")
        ctx.replace_inputs(node, [half_x, gate])
