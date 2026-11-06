"""Local patches for tf2onnx 1.17.0 in this venv (idempotent; re-run after any pip reinstall). From the x2e lane:

1. tflite_handlers/tfl_math.py: a TFLite ADD/SUB/MUL/DIV node without a `fused_activation_function` attribute
   (the zoo's ssdlite_mobilenetv3_320_fp16.tflite has them) raised KeyError. TFLite's default is NONE.
2. tflite_handlers/tfl_x2e_extra.py (copied from this folder's tfl_x2e_extra.py) + its import in tflite_handlers/__init__.py:
   handlers for TFL_GELU, TFL_RELU_0_TO_1, TFL_RELU_N1_TO_1, which tf2onnx otherwise leaves unconverted.

3. tflite_handlers/tfl_math.py, TFL_FULLY_CONNECTED: when keep_num_dims is false and the input has rank > 2, the
   handler inserts a Reshape and rewrites the node's inputs as [reshaped input, weights], dropping the bias (input 2),
   so the `len(node.input) == 3` branch never adds it (13 of the 52 zoo files lose 1 to 24 biases). The patch keeps
   the bias. ZOO_FC_BIAS_FIX=0 in the environment restores the old behaviour, to rebuild the files the table
   measured (convert_all.py sets it from recipe.json).

All three are upstream issue/PR candidates (onnx/tensorflow-onnx)."""

import os
import shutil
import sys

import tf2onnx

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(tf2onnx.__file__), "tflite_handlers")


def replace_once(path, old, new, label):
    src = open(path, encoding="utf-8").read()
    if new in src:
        print(f"already: {label}")
        return
    if old not in src:
        print(f"anchor not found ({label}); tf2onnx changed: {path}")
        sys.exit(1)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(src.replace(old, new, 1))
    os.replace(tmp, path)
    print(f"patched: {label}")


replace_once(
    os.path.join(PKG, "tfl_math.py"),
    """    activation_fn = node.attr['fused_activation_function'].s
    del node.attr['fused_activation_function']
""",
    """    # x2e workaround: a missing attribute means the TFLite default, NONE.
    if 'fused_activation_function' not in node.attr:
        return
    activation_fn = node.attr['fused_activation_function'].s
    del node.attr['fused_activation_function']
""",
    "tfl_math missing fused_activation_function",
)

src_mod = os.path.join(HERE, "tfl_x2e_extra.py")
dst_mod = os.path.join(PKG, "tfl_x2e_extra.py")
if not os.path.exists(dst_mod) or open(src_mod, "rb").read() != open(dst_mod, "rb").read():
    shutil.copyfile(src_mod, dst_mod)
    print("copied: tfl_x2e_extra.py")
else:
    print("already: tfl_x2e_extra.py")

replace_once(
    os.path.join(PKG, "__init__.py"),
    """    tfl_postprocess,
)""",
    """    tfl_postprocess,
    tfl_x2e_extra,
)""",
    "tflite_handlers import tfl_x2e_extra",
)

replace_once(
    os.path.join(PKG, "tfl_math.py"),
    """            ctx.replace_inputs(node, [reshape_node.output[0], node.input[1]])
""",
    """            # windows-zoo fix: keep the bias (input 2); ZOO_FC_BIAS_FIX=0 drops it as tf2onnx 1.17.0 does
            keep = None if __import__("os").environ.get("ZOO_FC_BIAS_FIX", "1") != "0" else 2
            ctx.replace_inputs(node, [reshape_node.output[0]] + node.input[1:keep])
""",
    "tfl_math FULLY_CONNECTED keeps its bias when keep_num_dims is false",
)
