"""A content hash of an ONNX graph that ignores what tf2onnx changes from one run to the next.

    python graph_signature.py <a.onnx> [<b.onnx>]

tf2onnx 1.17.0 does not write the same bytes twice for the same .tflite: initializer names (const_fold_opt__N), the
order of independent nodes and graph.doc_string (it holds the input path) change between runs. This hash keeps what
decides the outputs and drops the names: every tensor is named by its content (a weight by the sha256 of its element
type, shape and values; a graph input by its name; a node output by the hash of the node's domain, op type,
attributes and input hashes, Merkle style), and the graph hash covers the opset imports, the graph inputs and
outputs (name, element type, shape) and the hash of each graph output. Two files with the same graph hash compute
the same function with the same weights. With two files it prints whether they match and, if not, the op counts.
"""

import hashlib
import json
import sys
from collections import Counter

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _sha(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else repr(p).encode())
    return h.hexdigest()


def tensor_hash(t):
    a = numpy_helper.to_array(t)
    return _sha(str(a.dtype), a.shape, np.ascontiguousarray(a).tobytes())


def attr_text(a):
    v = helper.get_attribute_value(a)
    if isinstance(v, onnx.TensorProto):
        v = tensor_hash(v)
    elif isinstance(v, onnx.GraphProto):
        v = _sha(v.SerializeToString())
    elif isinstance(v, list) and v and isinstance(v[0], onnx.TensorProto):
        v = [tensor_hash(x) for x in v]
    elif isinstance(v, list) and v and isinstance(v[0], onnx.GraphProto):
        v = [_sha(x.SerializeToString()) for x in v]
    elif isinstance(v, bytes):
        v = v.hex()
    return f"{a.name}={v!r}"


def value_info(v):
    tt = v.type.tensor_type
    return (v.name, tt.elem_type, [d.dim_value if d.HasField("dim_value") else d.dim_param for d in tt.shape.dim])


def signature(path):
    m = onnx.load(path)
    g = m.graph
    tensor = {"": "none"}
    weights = set()
    for t in g.initializer:
        tensor[t.name] = "w:" + tensor_hash(t)
        weights.add(t.name)
    inputs = [i for i in g.input if i.name not in weights]
    for i in inputs:
        tensor[i.name] = "in:" + i.name
    ops = Counter()
    undefined = []
    for n in g.node:
        ops[f"{n.domain + '.' if n.domain else ''}{n.op_type}"] += 1
        ins = []
        for x in n.input:
            if x not in tensor:
                undefined.append(x)
            ins.append(tensor.get(x, "undefined"))
        h = _sha(n.domain, n.op_type, sorted(attr_text(a) for a in n.attribute), ins)
        for k, o in enumerate(n.output):
            if o:
                tensor[o] = f"n:{h}:{k}"
    io = {"opset": sorted((o.domain, o.version) for o in m.opset_import),
          "inputs": [value_info(i) for i in inputs], "outputs": [value_info(o) for o in g.output]}
    outs = [(o.name, tensor.get(o.name, "undefined")) for o in g.output]
    return {"graph_hash": _sha(io, outs), "nodes": len(g.node), "weights": len(g.initializer),
            "ops": dict(sorted(ops.items())), "io": io, "undefined_inputs": undefined[:5]}


def main():
    sigs = [signature(p) for p in sys.argv[1:3]]
    if len(sigs) == 1:
        print(json.dumps(sigs[0], indent=1))
        return
    a, b = sigs
    same = a["graph_hash"] == b["graph_hash"]
    print(f"graph_hash {'same' if same else 'DIFFERENT'}: {a['graph_hash'][:16]} vs {b['graph_hash'][:16]} "
          f"(nodes {a['nodes']} vs {b['nodes']}, weights {a['weights']} vs {b['weights']})")
    if not same:
        for k in sorted(set(a["ops"]) | set(b["ops"])):
            if a["ops"].get(k) != b["ops"].get(k):
                print(f"  {k}: {a['ops'].get(k, 0)} vs {b['ops'].get(k, 0)}")
        if a["io"] != b["io"]:
            print(f"  io: {a['io']} vs {b['io']}")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
