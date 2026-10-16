"""Read-only macOS camera inventory, supported formats and standard UVC controls."""

import sys

import usb.util

from . import macos_cameras, uvc_controls
from .camera_identity import normalize_usb_id


def inspect_cameras(*, index=None, camera_id=None, location=None, list_only=False):
    if sys.platform != "darwin":
        raise ValueError("camera-properties currently supports macOS only")
    warnings = []
    try:
        native = macos_cameras.cameras()
    except (OSError, AttributeError, ValueError) as exc:
        native = []
        warnings.append(f"AVFoundation discovery failed: {exc}")
    usb_devices = []
    try:
        try:
            usb_devices, usb_warnings = uvc_controls.devices()
            warnings.extend(usb_warnings)
        except (RuntimeError, OSError) as exc:
            warnings.append(f"USB control discovery unavailable: {exc}")
        records = [(dict(camera), None) for camera in native]
        for metadata, device, interfaces in usb_devices:
            # Match only the exact location+VID+PID representation used by USB
            # AVFoundation IDs; never infer identity from name or enumeration order.
            location_id = metadata["location_id"]
            native_id = (
                f"{location_id}{metadata['vendor_id']}{metadata['product_id']}"
                if location_id
                else None
            )
            matches = [
                r
                for r in records
                if native_id and (r[0].get("native_id") or "").lower() == native_id
            ]
            if len(matches) == 1:
                record = matches[0]
                record[0].update({k: v for k, v in metadata.items() if k != "name"})
                records[records.index(record)] = (record[0], (device, interfaces))
                record[0]["usb_match"] = "exact native ID encodes USB location, vendor and product"
            else:
                records.append(
                    (
                        {
                            **metadata,
                            "index": None,
                            "native_id": None,
                            "active_format": None,
                            "formats": [],
                            "usb_match": "no exact AVFoundation match; capture index unknown",
                        },
                        (device, interfaces),
                    )
                )
        if camera_id and camera_id.startswith("usb:"):
            camera_id = normalize_usb_id(camera_id)
        if location is not None:
            try:
                location = f"0x{int(location.split('/')[0].strip(), 16):08x}"
            except ValueError:
                raise ValueError("Use a hexadecimal USB location, for example 0x14200000") from None
        selected = [
            (camera, usb_device)
            for camera, usb_device in records
            if (index is None or camera.get("index") == index)
            and (camera_id is None or camera_id in (camera.get("native_id"), camera.get("usb_id")))
            and (location is None or camera.get("location_id") == location)
        ]
        if index is not None or camera_id is not None or location is not None:
            if not selected:
                raise ValueError(
                    "No matching camera. Run camera-properties --list to see selectors."
                )
            if len(selected) != 1:
                raise ValueError(
                    "Camera selector is ambiguous; use a unique native ID or location."
                )
        for camera, usb_device in selected:
            camera.setdefault("errors", [])
            camera["controls"] = []
            if list_only:
                camera.pop("formats", None)
                camera["controls_status"] = "not queried (--list)"
            elif usb_device:
                try:
                    camera["controls"] = uvc_controls.read_controls(*usb_device)
                    camera["controls_status"] = (
                        "standard UVC controls queried"
                        if camera["controls"]
                        else "no standard UVC controls advertised"
                    )
                except (ValueError, OSError) as exc:
                    camera["controls_status"] = "unavailable"
                    camera["errors"].append(str(exc))
            else:
                camera["controls_status"] = "unavailable: no exact USB UVC device match"
        return {
            "cameras": [camera for camera, _ in selected],
            "warnings": warnings,
            "notes": [
                "Read-only snapshot: no capture session, configuration changes or SET requests.",
                "active_format is AVFoundation device state, not a measured capture stream; "
                "fps_ranges are capabilities, not current measured FPS.",
                "Indices follow pinned OpenCV 4.12 AVFoundation ordering and may change after "
                "reconnection. Verify the physical preview before calibration.",
                "Unavailable values are null, not zero. Vendor extension controls are not decoded. "
                "Non-USB cameras may expose formats but no UVC controls.",
            ],
        }
    finally:
        for _, device, _ in usb_devices:
            usb.util.dispose_resources(device)


def format_report(report):
    lines = []
    for camera in report["cameras"]:
        lines.append(f"Camera {camera.get('index')} — {camera['name']}")
        for field in ("native_id", "usb_id", "location_id", "serial"):
            if camera.get(field):
                lines.append(f"  {field}: {camera[field]}")
        active = camera.get("active_format")
        if active:
            lines.append(
                f"  Active device format: {active['width_px']} × {active['height_px']} "
                f"({active['pixel_format']})"
            )
        formats = camera.get("formats", [])
        if formats:
            lines.append("  Supported formats (resolution, pixel format, FPS range):")
            for f in formats:
                fps = ", ".join(f"{r['min']:g}–{r['max']:g}" for r in f["fps_ranges"])
                lines.append(f"    {f['width_px']} × {f['height_px']} {f['pixel_format']}: {fps}")
        lines.append(f"  Controls: {camera['controls_status']}")
        for name, control in camera.get("native_controls", {}).items():
            modes = control["supported_modes"]
            value = control["value"]
            lines.append(
                f"    {name}: {modes.get(str(value), 'unavailable')} "
                f"(AVFoundation value={value}, supported={modes})"
            )
        for c in camera["controls"]:
            values = ", ".join(
                f"{k}={c.get(k)}" for k in ("value", "min", "max", "resolution", "default")
            )
            lines.append(
                f"    {c['name']} [interface {c['interface']}, unit {c['unit_id']}]: "
                f"{values}" + (f" ({c['unit']})" if c.get("unit") else "")
            )
            if c.get("disabled_by_auto_mode"):
                lines.append("      Disabled by automatic mode")
            for key, error in c["errors"].items():
                lines.append(f"      {key}: {error}")
        lines.extend(f"  Error: {e}" for e in camera["errors"])
    if not report["cameras"]:
        lines.append("No cameras found. Check connections and macOS camera permissions.")
    lines.extend(f"Warning: {w}" for w in report["warnings"])
    lines.extend(f"Note: {n}" for n in report["notes"])
    return "\n".join(lines)
