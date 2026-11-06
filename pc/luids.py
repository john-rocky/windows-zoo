"""Print the Windows counter LUID keys ('0xHIGH_0xLOW') of the Intel GPU and the NPU as JSON, from OpenVINO's
DEVICE_LUID. They change on every reboot, so the chain scripts read them here instead of keeping them."""

import json

import openvino as ov


def luid_key(device_luid):
    b = bytes.fromhex(device_luid)
    return f"0x{int.from_bytes(b[4:8], 'little'):08x}_0x{int.from_bytes(b[0:4], 'little'):08x}"


core = ov.Core()
out = {}
for dev in ("GPU", "NPU"):
    try:
        out[dev] = luid_key(str(core.get_property(dev, "DEVICE_LUID")))
    except Exception as e:  # noqa: BLE001
        out[dev + "_error"] = str(e)[:200]
print(json.dumps(out))
