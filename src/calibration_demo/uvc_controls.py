"""Read standard UVC camera/processing controls. Never claim, detach, configure or SET."""

import libusb_package
import usb.core
import usb.util

# Descriptor bitmap bit -> (selector, name, byte width, signed).
# USB Video Class 1.5, Camera Terminal and Processing Unit control tables.
CAMERA_CONTROLS = {
    0: (1, "scanning_mode", 1, False),
    1: (2, "auto_exposure_mode", 1, False),
    2: (3, "auto_exposure_priority", 1, False),
    3: (4, "exposure_absolute", 4, False),
    4: (5, "exposure_relative", 1, True),
    5: (6, "focus_absolute", 2, False),
    6: (7, "focus_relative", 2, False),
    7: (9, "iris_absolute", 2, False),
    8: (10, "iris_relative", 1, True),
    9: (11, "zoom_absolute", 2, False),
    10: (12, "zoom_relative", 3, False),
    11: (13, "pan_tilt_absolute", 8, False),
    12: (14, "pan_tilt_relative", 4, False),
    13: (15, "roll_absolute", 2, True),
    14: (16, "roll_relative", 2, False),
    17: (8, "autofocus", 1, False),
    18: (17, "privacy", 1, False),
    19: (18, "focus_simple", 1, False),
    20: (19, "window", 12, False),
    21: (20, "region_of_interest", 10, False),
}
PROCESSING_CONTROLS = {
    0: (2, "brightness", 2, True),
    1: (3, "contrast", 2, False),
    2: (6, "hue", 2, True),
    3: (7, "saturation", 2, False),
    4: (8, "sharpness", 2, False),
    5: (9, "gamma", 2, False),
    6: (10, "white_balance_temperature", 2, False),
    7: (12, "white_balance_component", 4, False),
    8: (1, "backlight_compensation", 2, False),
    9: (4, "gain", 2, False),
    10: (5, "power_line_frequency", 1, False),
    11: (16, "auto_hue", 1, False),
    12: (11, "auto_white_balance_temperature", 1, False),
    13: (13, "auto_white_balance_component", 1, False),
    14: (14, "digital_multiplier", 2, False),
    15: (15, "digital_multiplier_limit", 2, False),
    16: (17, "analog_video_standard", 1, False),
    17: (18, "analog_video_lock_status", 1, False),
    18: (19, "auto_contrast", 1, False),
}
COMPOUND = {
    "focus_relative",
    "zoom_relative",
    "pan_tilt_relative",
    "roll_relative",
    "window",
    "region_of_interest",
}


def descriptor_controls(extra):
    """Yield unit/selector metadata only for controls advertised in descriptor bitmaps."""
    offset = 0
    while offset < len(extra):
        length = extra[offset]
        if length < 3 or offset + length > len(extra):
            raise ValueError("Malformed UVC class descriptor")
        d = extra[offset : offset + length]
        offset += length
        if d[1] != 0x24:
            continue
        if d[2] == 2 and len(d) >= 15 and int.from_bytes(d[4:6], "little") == 0x0201:
            table, size, start = CAMERA_CONTROLS, d[14], 15
        elif d[2] == 5 and len(d) >= 8:
            table, size, start = PROCESSING_CONTROLS, d[7], 8
        else:
            continue
        if start + size > len(d):
            raise ValueError("Truncated UVC control bitmap")
        bitmap = int.from_bytes(d[start : start + size], "little")
        for bit, (selector, name, width, signed) in table.items():
            if bitmap & (1 << bit):
                yield d[3], selector, name, width, signed


def location_id(device):
    """Darwin's locationID: bus byte followed by up to six port-path nibbles."""
    ports = device.port_numbers
    if device.bus is None or not ports or len(ports) > 6 or any(p > 15 for p in ports):
        return None
    value = device.bus << 24
    for i, port in enumerate(ports):
        value |= port << (20 - i * 4)
    return f"0x{value:08x}"


def devices():
    backend = libusb_package.get_libusb1_backend()
    if backend is None:
        raise RuntimeError("The packaged libusb backend could not be loaded")
    result, warnings = [], []
    for device in usb.core.find(find_all=True, backend=backend):
        try:
            # Reading configuration descriptors does not activate a configuration.
            interfaces = [
                interface
                for configuration in device
                for interface in configuration
                if interface.bInterfaceClass == 14
                and interface.bInterfaceSubClass == 1
                and interface.bAlternateSetting == 0
            ]
            if not interfaces:
                continue
            errors = []

            def string(attribute):
                try:
                    return getattr(device, attribute) or None
                except (usb.core.USBError, ValueError) as exc:
                    errors.append(f"{attribute}: {exc}")
                    return None

            serial, name = string("serial_number"), string("product")
            result.append(
                (
                    {
                        "name": name or f"USB camera {device.idVendor:04x}:{device.idProduct:04x}",
                        "usb_id": f"usb:{device.idVendor:04x}:{device.idProduct:04x}:{serial}"
                        if serial and not any(c.isspace() or c == ":" for c in serial)
                        else None,
                        "vendor_id": f"{device.idVendor:04x}",
                        "product_id": f"{device.idProduct:04x}",
                        "location_id": location_id(device),
                        "serial": serial,
                        "errors": errors,
                    },
                    device,
                    interfaces,
                )
            )
        except (usb.core.USBError, ValueError) as exc:
            warnings.append(f"USB {device.idVendor:04x}:{device.idProduct:04x}: {exc}")
    return result, warnings


def _decode(name, data, signed):
    if name == "pan_tilt_absolute":
        return {
            "pan_arcseconds": int.from_bytes(data[:4], "little", signed=True),
            "tilt_arcseconds": int.from_bytes(data[4:], "little", signed=True),
        }
    if name == "white_balance_component":
        return {
            "blue": int.from_bytes(data[:2], "little"),
            "red": int.from_bytes(data[2:], "little"),
        }
    if name in COMPOUND:
        return {"raw_hex": data.hex()}
    return int.from_bytes(data, "little", signed=signed)


def read_controls(device, interfaces, timeout_ms=300):
    controls = []
    # GET_CONFIGURATION only; never select/activate an inactive configuration.
    active = device.get_active_configuration().index
    interfaces = [interface for interface in interfaces if interface.configuration == active]
    for interface in interfaces:
        for unit, selector, name, width, signed in descriptor_controls(interface.extra_descriptors):
            item = {
                "name": name,
                "unit_id": unit,
                "interface": interface.bInterfaceNumber,
                "selector": selector,
                "source": "USB UVC",
                "errors": {},
            }
            if name == "exposure_absolute":
                item["unit"] = "100 microseconds"
            if name == "auto_exposure_mode":
                item["modes"] = {
                    "1": "manual",
                    "2": "auto",
                    "4": "shutter priority",
                    "8": "aperture priority",
                }
            for key, request, size in (
                ("info", 0x86, 1),
                ("value", 0x81, width),
                ("min", 0x82, width),
                ("max", 0x83, width),
                ("resolution", 0x84, width),
                ("default", 0x87, width),
            ):
                try:
                    # Class/interface/device-to-host only. Do not use PyUSB's implicit
                    # interface claim (which can select a configuration): recipient 1
                    # requests are claimed automatically by PyUSB; use backend directly.
                    device._ctx.managed_open()
                    buffer = usb.util.create_buffer(size)
                    count = device._ctx.backend.ctrl_transfer(
                        device._ctx.handle,
                        0xA1,
                        request,
                        selector << 8,
                        (unit << 8) | interface.bInterfaceNumber,
                        buffer,
                        timeout_ms,
                    )
                    if count != size:
                        raise ValueError(f"Short read: expected {size}, received {count}")
                    data = bytes(buffer)
                    item[key] = data[0] if key == "info" else _decode(name, data, signed)
                except (usb.core.USBError, ValueError) as exc:
                    item[key] = None
                    item["errors"][key] = str(exc)
            item["status"] = "ok" if item["value"] is not None else "unavailable"
            if item["info"] is not None:
                item["readable"] = bool(item["info"] & 1)
                item["writable"] = bool(item["info"] & 2)
                item["disabled_by_auto_mode"] = bool(item["info"] & 4)
            controls.append(item)
    return controls
