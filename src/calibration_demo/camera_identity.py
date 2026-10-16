"""USB device identity for session provenance; camera index is not an identity."""

import json
import re
import subprocess
import sys


def normalize_usb_id(value):
    match = re.fullmatch(r"usb:([0-9a-fA-F]{4}):([0-9a-fA-F]{4}):([^:\s]+)", value.strip())
    if not match:
        raise ValueError("Use usb:VID:PID:SERIAL, for example usb:046d:0825:AB12CD34")
    vendor, product, serial = match.groups()
    return f"usb:{vendor.lower()}:{product.lower()}:{serial}"


def usb_devices():
    """List physical USB devices with serials. This does not map them to OpenCV indices."""
    if sys.platform != "darwin":
        return []
    result = subprocess.run(
        ["system_profiler", "SPUSBDataType", "-json"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    roots = json.loads(result.stdout).get("SPUSBDataType", [])
    devices = []

    def visit(node):
        if not isinstance(node, dict):
            return
        vendor = re.search(r"0x([0-9a-fA-F]{4})", node.get("vendor_id", ""))
        product = re.search(r"0x([0-9a-fA-F]{4})", node.get("product_id", ""))
        serial = node.get("serial_num")
        if vendor and product and serial:
            name = node.get("_name", "").strip()
            if name == "miscellaneous_device":
                supplier = node.get("vendor_id", "").split("(", 1)
                name = supplier[1].rstrip(")") if len(supplier) > 1 else name
            devices.append(
                {
                    "id": normalize_usb_id(f"usb:{vendor.group(1)}:{product.group(1)}:{serial}"),
                    "name": name,
                    "location_id": node.get("location_id"),
                }
            )
        for child in node.get("_items", []):
            visit(child)

    for root in roots:
        visit(root)
    return devices


def recorded_id(payload):
    if payload.get("camera_identity_source") == "missing":
        return None
    values = payload.get("camera_identities", [])
    return normalize_usb_id(values[0]) if len(values) == 1 and values[0] else None


def suggested_usb_camera(devices):
    """Suggest one serial-bearing non-hub USB device; the user must verify the preview."""
    candidates = [device["id"] for device in devices if "hub" not in device.get("name", "").lower()]
    return candidates[0] if len(candidates) == 1 else None
