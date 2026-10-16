import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .calibration import Board
from .camera_identity import normalize_usb_id, recorded_id, usb_devices
from .geometry import Rig, measure, project


def synthetic(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    k = np.array([[1000.0, 0, 640], [0, 1000, 360], [0, 0, 1]])
    r = cv2.Rodrigues(np.array([0.0, 0.12, 0.0]))[0]
    rig = Rig(
        k,
        np.array([-0.05, 0.01, 0.001, -0.001, 0.0]),
        k.copy(),
        np.zeros(5),
        r,
        (-r @ np.array([1.0, 0.0, 0.0])).reshape(3, 1),
        (1280, 720),
        {"synthetic": True},
    )
    xyz = np.array([[-0.5, -0.3, 4.0], [0.7, 0.4, 5.0]])
    points = [project(rig, xyz, i) for i in (0, 1)]
    for i, name in enumerate(("left.png", "right.png")):
        im = np.full((720, 1280, 3), (38, 31, 25), np.uint8)
        cv2.putText(
            im,
            "SYNTHETIC REHEARSAL | NO REAL CAMERA DATA",
            (45, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (210, 210, 210),
            2,
        )
        for label, p in zip(("A", "B"), points[i]):
            x, y = np.rint(p).astype(int)
            cv2.drawMarker(im, (x, y), (255, 190, 0), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(
                im, label, (x + 18, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
        cv2.line(
            im,
            tuple(np.rint(points[i][0]).astype(int)),
            tuple(np.rint(points[i][1]).astype(int)),
            (100, 100, 100),
            1,
        )
        cv2.imwrite(str(directory / name), im)
    rig.save(directory / "calibration.json")
    (directory / "session.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                points=[p.tolist() for p in points],
                reference_m=float(np.linalg.norm(xyz[1] - xyz[0])),
                host_grab_times=[],
                note="Synthetic target; verifies software geometry, not hardware accuracy",
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Two-camera calibration and 3D measurement")
    sub = parser.add_subparsers(dest="command", required=True)
    gui = sub.add_parser("gui")
    gui.add_argument("--session")
    gui.add_argument(
        "--single-camera", action="store_true", help="Calibrate one camera's intrinsics"
    )
    gui.add_argument("--camera", type=int, default=0, help="Left/single camera index (default: 0)")
    gui.add_argument("--screenshot", help=argparse.SUPPRESS)
    sub.add_parser("sequential", help="Offline-first one-camera/two-position stereo workflow")
    board = sub.add_parser("board")
    board.add_argument("--output", default="charuco.png")
    board.add_argument("--columns", type=int, default=7)
    board.add_argument("--rows", type=int, default=5)
    board.add_argument("--square-mm", type=float, default=40)
    board.add_argument("--marker-mm", type=float)
    board.add_argument("--dictionary", default="DICT_4X4_100")
    synth = sub.add_parser("synthetic")
    synth.add_argument("--output", default="examples/synthetic")
    replay = sub.add_parser("measure")
    replay.add_argument("session")
    review = sub.add_parser("review", help="Inspect and refit saved single-camera images")
    review.add_argument(
        "session", nargs="?", help="Session folder (default: latest with intrinsics)"
    )
    sub.add_parser("camera-id", help="List USB vendor/product/serial identifiers on this Mac")
    tag = sub.add_parser("tag-session", help="Record a verified camera ID in one saved session")
    tag.add_argument("session")
    tag.add_argument("usb_id", help="usb:VID:PID:SERIAL from camera-id")
    tag.add_argument(
        "--verified-from-preview",
        action="store_true",
        help="Confirm the session's preview used this physical camera",
    )
    compare = sub.add_parser(
        "compare-sessions", help="Check repeatability of two intrinsic sessions"
    )
    compare.add_argument("first")
    compare.add_argument("second")
    compare.add_argument("--repeats", type=int, default=2)
    compare.add_argument("--model", choices=("4", "5", "8"), default="5")
    compare.add_argument("--json", action="store_true")
    mix = sub.add_parser("mix-sessions", help="Fit one calibration from two image sets")
    mix.add_argument("first")
    mix.add_argument("second")
    mix.add_argument("--output", required=True)
    mix.add_argument("--model", choices=("4", "5", "8"), default="5")
    mix.add_argument(
        "--assume-same-camera",
        action="store_true",
        help="Permit missing recorded identity after manual camera verification",
    )
    args = parser.parse_args()
    if args.command == "gui":
        from .gui import launch

        raise SystemExit(launch(args.session, args.screenshot, args.single_camera, args.camera))
    if args.command == "sequential":
        from .offline_gui import launch

        raise SystemExit(launch())
    if args.command == "board":
        Board(
            squares_x=args.columns,
            squares_y=args.rows,
            square_m=args.square_mm / 1000,
            marker_m=(args.marker_mm if args.marker_mm is not None else args.square_mm * 0.7)
            / 1000,
            dictionary=args.dictionary,
        ).write(args.output)
        print(
            f"Board area: {args.square_mm * args.columns:g} × "
            f"{args.square_mm * args.rows:g} mm. Verify physical square length."
        )
    if args.command == "synthetic":
        synthetic(args.output)
    if args.command == "measure":
        path = Path(args.session)
        obj = json.loads((path / "session.json").read_text())
        print(
            json.dumps(
                measure(
                    Rig.load(path / "calibration.json"), *obj["points"], obj.get("reference_m")
                ),
                indent=2,
            )
        )
    if args.command == "review":
        from .review_gui import launch

        session = args.session
        if session is None:
            candidates = sorted(Path("sessions").glob("*/intrinsics.json"))
            if not candidates:
                parser.error("No intrinsic session found; pass a session folder")
            session = str(candidates[-1].parent)
        raise SystemExit(launch(session))
    if args.command == "camera-id":
        print(json.dumps(usb_devices(), indent=2))
    if args.command == "tag-session":
        if not args.verified_from_preview:
            parser.error("Verify the saved preview, then pass --verified-from-preview")
        source = Path(args.session) / "intrinsics.json"
        obj = json.loads(source.read_text())
        if len(obj["cameras"]) != 1:
            parser.error("tag-session currently supports one-camera sessions")
        value = normalize_usb_id(args.usb_id)
        if recorded_id(obj) and recorded_id(obj) != value:
            parser.error("This session already has a different camera ID")
        if value not in {device["id"] for device in usb_devices()}:
            parser.error("That USB ID is not connected to this Mac")
        obj["camera_identities"] = [value]
        obj["camera_identity_source"] = "manual_after_capture"
        temporary = source.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(obj, indent=2))
        temporary.replace(source)
        print(f"Recorded {value} in {source}")
    if args.command == "compare-sessions":
        from .sessions import compare as compare_sessions

        try:
            report = compare_sessions(args.first, args.second, args.repeats, args.model)
        except (ValueError, FileNotFoundError, cv2.error) as exc:
            parser.error(str(exc))
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"Camera identity: {report['camera_identity']}")
            print(f"Model: {report['model']}")
            for label in ("first", "second"):
                item = report[label]
                print(
                    f"{label}: {item['selected_images']} views, fit {item['fit_rms_px']:.3f} px, "
                    f"held-out {item['held_out_rms_px']:.3f} px, "
                    f"split ray median/max {item['split_ray_median_px']:.1f}/"
                    f"{item['split_ray_max_px']:.1f} px, "
                    f"well-sampled cells {item['coverage_cells_with_10_corners']}/48, "
                    f"repeat deterministic {item['repeat_deterministic_1e_9']}, "
                    f"input SHA-256 {item['input_digest_sha256'][:12]}"
                )
            print(
                f"Between sessions: fx {report['fx_difference_percent']:+.2f}%, "
                f"fy {report['fy_difference_percent']:+.2f}%, "
                f"cx {report['cx_difference_px']:+.1f} px, "
                f"cy {report['cy_difference_px']:+.1f} px, "
                f"ray median/max {report['between_session_ray_median_px']:.1f}/"
                f"{report['between_session_ray_max_px']:.1f} px"
            )
            print(f"Identical image files in both sessions: {report['shared_image_count']}")
    if args.command == "mix-sessions":
        from .sessions import mix as mix_sessions

        try:
            result = mix_sessions(
                args.first, args.second, args.output, args.assume_same_camera, args.model
            )
        except (ValueError, FileNotFoundError, FileExistsError, cv2.error) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
