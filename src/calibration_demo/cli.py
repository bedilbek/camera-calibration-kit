import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .calibration import Board
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
    gui.add_argument("--screenshot", help=argparse.SUPPRESS)
    board = sub.add_parser("board")
    board.add_argument("--output", default="charuco.png")
    board.add_argument("--square-mm", type=float, default=40)
    synth = sub.add_parser("synthetic")
    synth.add_argument("--output", default="examples/synthetic")
    replay = sub.add_parser("measure")
    replay.add_argument("session")
    args = parser.parse_args()
    if args.command == "gui":
        from .gui import launch

        raise SystemExit(launch(args.session, args.screenshot))
    if args.command == "board":
        Board(square_m=args.square_mm / 1000, marker_m=args.square_mm * 0.7 / 1000).write(
            args.output
        )
        print(
            f"Board area: {args.square_mm * 7:g} × {args.square_mm * 5:g} mm. Verify physical square length."
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


if __name__ == "__main__":
    main()
