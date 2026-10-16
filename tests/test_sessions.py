"""Hardware identity guards for cross-session calibration."""

from types import SimpleNamespace

import pytest

from calibration_demo import sessions
from calibration_demo.camera_identity import normalize_usb_id, recorded_id, suggested_usb_camera


def test_usb_id_is_normalized_and_requires_serial():
    assert normalize_usb_id("usb:046D:0825:AB12CD34") == "usb:046d:0825:AB12CD34"
    with pytest.raises(ValueError, match="SERIAL"):
        normalize_usb_id("usb:046d:0825")


@pytest.mark.parametrize(
    ("first_id", "second_id", "message"),
    [
        ("usb:046d:0825:one", "usb:046d:0825:two", "differ"),
        (None, None, "missing"),
    ],
)
def test_mix_blocks_conflicting_or_missing_identity(
    monkeypatch, tmp_path, first_id, second_id, message
):
    first = SimpleNamespace(saved={"camera_identities": [first_id] if first_id else []})
    second = SimpleNamespace(saved={"camera_identities": [second_id] if second_id else []})
    monkeypatch.setattr(sessions, "_load_pair", lambda _a, _b: (first, second))
    with pytest.raises(ValueError, match=message):
        sessions.mix("one", "two", tmp_path / "mix")
    assert not (tmp_path / "mix").exists()


def test_recorded_id_requires_one_camera():
    assert recorded_id({"camera_identities": ["usb:046D:0825:serial"]}) == ("usb:046d:0825:serial")
    assert recorded_id({"camera_identities": [None]}) is None
    assert recorded_id({"camera_identities": []}) is None
    assert (
        recorded_id(
            {"camera_identities": ["usb:046d:0825:AB12CD34"], "camera_identity_source": "missing"}
        )
        is None
    )


def test_usb_suggestion_requires_one_non_hub_device():
    hub = {"id": "usb:17ef:721c:HUB", "name": "Lenovo Powered Hub"}
    webcam = {"id": "usb:046d:0825:CAMERA", "name": "Logitech Inc."}
    other = {"id": "usb:1234:5678:OTHER", "name": "USB storage"}
    assert suggested_usb_camera([hub, webcam]) == webcam["id"]
    assert suggested_usb_camera([hub, webcam, other]) is None


def test_mix_preserves_source_folder(monkeypatch, tmp_path):
    payload = {"camera_identities": ["usb:046d:0825:AB12CD34"]}
    first = SimpleNamespace(saved=payload, root=tmp_path / "a")
    second = SimpleNamespace(saved=payload, root=tmp_path / "b")
    monkeypatch.setattr(sessions, "_load_pair", lambda _a, _b: (first, second))
    with pytest.raises(ValueError, match="outside both source"):
        sessions.mix("a", "b", first.root / "mix")
