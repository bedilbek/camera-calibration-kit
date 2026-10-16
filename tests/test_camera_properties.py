"""Hardware-independent checks for camera selection and read-only USB requests."""

from types import SimpleNamespace

import pytest
import usb.core

from calibration_demo import camera_properties as properties
from calibration_demo import uvc_controls as uvc


def processing_descriptor(bitmap=1):
    return bytes([11, 0x24, 5, 3, 1, 0, 0, 2, bitmap, 0, 0])


def test_descriptor_bitmaps_use_selectors_not_bit_numbers():
    camera = bytes([18, 0x24, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 8, 0, 2])
    assert list(uvc.descriptor_controls(camera + processing_descriptor())) == [
        (1, 4, "exposure_absolute", 4, False),
        (1, 8, "autofocus", 1, False),
        (3, 2, "brightness", 2, True),
    ]


@pytest.mark.parametrize(
    "descriptor", [b"\x00\x24\x05", b"\x20\x24\x05", bytes([8, 0x24, 5, 3, 1, 0, 0, 2])]
)
def test_rejects_malformed_descriptor_without_looping(descriptor):
    with pytest.raises(ValueError, match="Malformed|Truncated"):
        list(uvc.descriptor_controls(descriptor))


def test_signed_values_and_read_only_transfer_addressing():
    calls = []

    def transfer(handle, request_type, request, value, index, buffer, timeout):
        calls.append((request_type, request, value, index))
        data = b"\x03" if request == 0x86 else (-10).to_bytes(2, "little", signed=True)
        for i, value in enumerate(data):
            buffer[i] = value
        return len(data)

    device = SimpleNamespace(
        get_active_configuration=lambda: SimpleNamespace(index=0),
        _ctx=SimpleNamespace(
            managed_open=lambda: None,
            handle=object(),
            backend=SimpleNamespace(ctrl_transfer=transfer),
        ),
    )
    interface = SimpleNamespace(
        configuration=0, bInterfaceNumber=2, extra_descriptors=processing_descriptor()
    )
    control = uvc.read_controls(device, [interface])[0]
    assert control["value"] == -10
    assert control["readable"] and control["writable"]
    assert not control["disabled_by_auto_mode"]
    assert control["status"] == "ok"
    assert all(
        kind == 0xA1 and value == 0x0200 and index == 0x0302 for kind, _, value, index in calls
    )
    assert {request for _, request, _, _ in calls} == {0x81, 0x82, 0x83, 0x84, 0x86, 0x87}


@pytest.mark.parametrize("failure", ["stall", "short", "zero"])
def test_unavailable_is_distinct_from_valid_zero(failure):
    def transfer(handle, kind, request, value, index, buffer, timeout):
        if failure == "stall":
            raise usb.core.USBError("Pipe stalled")
        return 0 if failure == "short" else len(buffer)

    device = SimpleNamespace(
        get_active_configuration=lambda: SimpleNamespace(index=0),
        _ctx=SimpleNamespace(
            managed_open=lambda: None, handle=None, backend=SimpleNamespace(ctrl_transfer=transfer)
        ),
    )
    interface = SimpleNamespace(
        configuration=0, bInterfaceNumber=0, extra_descriptors=processing_descriptor()
    )
    control = uvc.read_controls(device, [interface])[0]
    assert control["value"] == (0 if failure == "zero" else None)
    assert bool(control["errors"]) == (failure != "zero")


def test_compound_controls_preserve_signed_axes():
    data = (-3600).to_bytes(4, "little", signed=True) + (7200).to_bytes(4, "little", signed=True)
    assert uvc._decode("pan_tilt_absolute", data, False) == {
        "pan_arcseconds": -3600,
        "tilt_arcseconds": 7200,
    }


def test_darwin_location_includes_hub_path():
    assert uvc.location_id(SimpleNamespace(bus=0x14, port_numbers=(2, 3))) == "0x14230000"
    assert uvc.location_id(SimpleNamespace(bus=0x14, port_numbers=None)) is None


@pytest.fixture
def inventory(monkeypatch):
    monkeypatch.setattr(properties.sys, "platform", "darwin")
    native = [
        {"index": 0, "native_id": "0x14200000046d0825", "name": "Webcam", "formats": []},
        {"index": 1, "native_id": "continuity-id", "name": "Phone", "formats": []},
    ]
    metadata = {
        "usb_id": "usb:046d:0825:ABC",
        "location_id": "0x14200000",
        "vendor_id": "046d",
        "product_id": "0825",
        "name": "USB name",
        "errors": [],
    }
    device = object()
    calls, releases = [], []
    monkeypatch.setattr(properties.macos_cameras, "cameras", lambda: native)
    monkeypatch.setattr(uvc, "devices", lambda: ([(metadata, device, [])], []))
    monkeypatch.setattr(uvc, "read_controls", lambda *args: calls.append(args) or [])
    monkeypatch.setattr(properties.usb.util, "dispose_resources", releases.append)
    return native, metadata, calls, releases, device


@pytest.mark.parametrize(
    "selector",
    [
        {"index": 0},
        {"camera_id": "usb:046D:0825:ABC"},
        {"camera_id": "0x14200000046d0825"},
        {"location": "0x14200000 / 2"},
    ],
)
def test_selectors_merge_exact_usb_device_and_release(inventory, selector):
    _, _, calls, releases, device = inventory
    report = properties.inspect_cameras(**selector)
    assert len(report["cameras"]) == 1
    assert report["cameras"][0]["name"] == "Webcam"
    assert report["cameras"][0]["usb_id"] == "usb:046d:0825:ABC"
    assert calls == [(device, [])]
    assert releases == [device]


def test_unselected_cameras_and_list_do_not_read_controls(inventory):
    _, _, calls, _, _ = inventory
    properties.inspect_cameras(index=1)
    properties.inspect_cameras(list_only=True)
    assert calls == []


def test_missing_selector_fails_and_releases(inventory):
    _, _, _, releases, device = inventory
    with pytest.raises(ValueError, match="No matching"):
        properties.inspect_cameras(index=99)
    assert releases == [device]


def test_unmatched_usb_is_never_assigned_a_capture_index(inventory):
    native, _, _, _, _ = inventory
    native[0]["native_id"] = "unknown-id"
    camera = properties.inspect_cameras(camera_id="usb:046d:0825:ABC")["cameras"][0]
    assert camera["index"] is None
    assert camera["native_id"] is None


def test_ambiguous_usb_serial_fails(inventory, monkeypatch):
    _, metadata, _, _, device = inventory
    monkeypatch.setattr(
        uvc,
        "devices",
        lambda: (
            [(metadata, device, []), ({**metadata, "location_id": "0x14300000"}, object(), [])],
            [],
        ),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        properties.inspect_cameras(camera_id=metadata["usb_id"])


def test_usb_backend_failure_preserves_native_inventory(inventory, monkeypatch):
    def fail():
        raise RuntimeError("backend missing")

    monkeypatch.setattr(uvc, "devices", fail)
    report = properties.inspect_cameras()
    assert len(report["cameras"]) == 2
    assert "backend missing" in report["warnings"][0]
