"""Read AVFoundation device metadata without opening a capture session or compiling code."""

import ctypes as ct
import re


class _Dimensions(ct.Structure):
    _fields_ = [("width", ct.c_int32), ("height", ct.c_int32)]


class _ObjC:
    def __init__(self):
        self.objc = ct.CDLL("/usr/lib/libobjc.A.dylib")
        self.av = ct.CDLL("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
        self.cm = ct.CDLL("/System/Library/Frameworks/CoreMedia.framework/CoreMedia")
        self.objc.objc_getClass.argtypes = [ct.c_char_p]
        self.objc.objc_getClass.restype = ct.c_void_p
        self.objc.sel_registerName.argtypes = [ct.c_char_p]
        self.objc.sel_registerName.restype = ct.c_void_p
        self.cm.CMVideoFormatDescriptionGetDimensions.argtypes = [ct.c_void_p]
        self.cm.CMVideoFormatDescriptionGetDimensions.restype = _Dimensions
        self.cm.CMFormatDescriptionGetMediaSubType.argtypes = [ct.c_void_p]
        self.cm.CMFormatDescriptionGetMediaSubType.restype = ct.c_uint32

    def send(self, obj, selector, *args, result=ct.c_void_p):
        # Each signature gets its own callable; never change a shared objc_msgSend signature.
        call = ct.CFUNCTYPE(result, ct.c_void_p, ct.c_void_p, *([ct.c_void_p] * len(args)))(
            ("objc_msgSend", self.objc)
        )
        return call(obj, self.objc.sel_registerName(selector.encode()), *args)

    def cls(self, name):
        return self.objc.objc_getClass(name.encode())

    def string(self, obj):
        value = self.send(obj, "UTF8String", result=ct.c_char_p)
        return value.decode("utf-8", errors="replace") if value else ""

    def array(self, obj):
        return [
            self.send(obj, "objectAtIndex:", i)
            for i in range(self.send(obj, "count", result=ct.c_ulong))
        ]

    def format(self, obj):
        desc = self.send(obj, "formatDescription")
        dims = self.cm.CMVideoFormatDescriptionGetDimensions(desc)
        subtype = self.cm.CMFormatDescriptionGetMediaSubType(desc)
        return {
            "width_px": dims.width,
            "height_px": dims.height,
            "pixel_format": subtype.to_bytes(4, "big").decode("ascii", errors="replace"),
            "fps_ranges": [
                {
                    "min": self.send(r, "minFrameRate", result=ct.c_double),
                    "max": self.send(r, "maxFrameRate", result=ct.c_double),
                }
                for r in self.array(self.send(obj, "videoSupportedFrameRateRanges"))
            ],
        }

    def modes(self, device):
        controls = {}
        for name, getter, support, labels in (
            (
                "focus_mode",
                "focusMode",
                "isFocusModeSupported:",
                ["locked", "auto", "continuous auto"],
            ),
            (
                "exposure_mode",
                "exposureMode",
                "isExposureModeSupported:",
                ["locked", "auto", "continuous auto"],
            ),
            (
                "white_balance_mode",
                "whiteBalanceMode",
                "isWhiteBalanceModeSupported:",
                ["locked", "auto", "continuous auto"],
            ),
        ):
            supported = {
                str(i): label
                for i, label in enumerate(labels)
                if self.send(device, support, i, result=ct.c_bool)
            }
            controls[name] = {
                "value": self.send(device, getter, result=ct.c_long) if supported else None,
                "supported_modes": supported,
                "source": "AVFoundation",
            }
        return controls


def cameras():
    """Match the pinned OpenCV 4.12 macOS enumeration: video + muxed, sorted by uniqueID."""
    api = _ObjC()
    pool = api.send(api.send(api.cls("NSAutoreleasePool"), "alloc"), "init")
    try:
        devices = []
        for media in ("AVMediaTypeVideo", "AVMediaTypeMuxed"):
            media_type = ct.c_void_p.in_dll(api.av, media).value
            devices.extend(
                api.array(api.send(api.cls("AVCaptureDevice"), "devicesWithMediaType:", media_type))
            )
        devices.sort(key=lambda d: api.string(api.send(d, "uniqueID")))
        result = [
            {
                "index": index,
                "native_id": api.string(api.send(device, "uniqueID")),
                "name": api.string(api.send(device, "localizedName")),
                "model": api.string(api.send(device, "modelID")),
                "active_format": api.format(api.send(device, "activeFormat")),
                "formats": [api.format(f) for f in api.array(api.send(device, "formats"))],
                "native_controls": api.modes(device),
            }
            for index, device in enumerate(devices)
        ]
        for camera in result:
            match = re.fullmatch(
                r"0x([0-9a-fA-F]{8})([0-9a-fA-F]{4})([0-9a-fA-F]{4})", camera["native_id"]
            )
            if match:
                camera["location_id"] = f"0x{match[1].lower()}"
                camera["location_source"] = "encoded AVFoundation native ID"
        return result
    finally:
        api.send(pool, "drain", result=None)
