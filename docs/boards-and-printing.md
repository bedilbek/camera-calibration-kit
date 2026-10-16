# Boards and Printing

## Available PDFs

The four one-page PDFs in [`patterns/`](../patterns/) were checked for landscape A4/A3 page dimensions, margins, and computer-vision detection. In each ChArUco filename, “5x7” means **5 rows and 7 columns of squares**. The app's **Board columns** and **Rows** fields should therefore read **7** and **5**.

| PDF | Pattern geometry | App support |
|---|---|---|
| [A4 ChArUco](../patterns/calib.io_CHARUCO_A4_297mmx210mm_5x7_30mm_22mm_DICT_4X4_legacy_off_start_id_0.pdf) | 7×5 squares; 30 mm square; 22 mm marker; 210×150 mm pattern | Yes; use `DICT_4X4_100`. |
| [A3 ChArUco](../patterns/calib.io_CHARUCO_A3_420mmx297mm_5x7_50mm_37mm_DICT_4X4_legacy_off_start_id_0.pdf) | 7×5 squares; 50 mm square; 37 mm marker; 350×250 mm pattern | Yes; use `DICT_4X4_100`. |
| [A4 checkerboard](../patterns/calib.io_CHECKERBOARD_A4_297mmx210mm_7x10_25mm.pdf) | 10×7 squares; 25 mm square; 9×6 inner corners | The current GUI does not fit plain checkerboards. |
| [A3 checkerboard](../patterns/calib.io_CHECKERBOARD_A3_420mmx297mm_7x10_35mm.pdf) | 10×7 squares; 35 mm square; 9×6 inner corners | The current GUI does not fit plain checkerboards. |

Both ChArUco PDFs decoded marker IDs 0–16 and all 24 inner corners with this project's OpenCV dictionary. Both checkerboards independently yielded 54 inner corners. This verifies digital compatibility; print accuracy still depends on your printer and mounting.

## Print and configure

Start with the **A4 ChArUco** PDF. Print at **100% / Actual Size**, with no “Fit to page” scaling. Measure several squares in both directions; each should be 30 mm. Keep the white margin and mount the whole sheet flat on a rigid support. If the print is stretched or curled, reprint it. The A3 version is useful when you need a larger board in the image at the same camera distance.

Before the first capture, enter **Board columns 7, Rows 5, Square 30 mm, Marker 22 mm, Dictionary `DICT_4X4_100`** for A4. For A3, change Square to **50 mm** and Marker to **37 mm**. Changing board parameters after collecting observations requires **Reset samples** and a fresh capture set. The app also accepts other ChArUco square counts and dictionaries when all five settings match the printed target. `Save board PNG` generates a separate OpenCV image; the calibrated, paper-sized PDF is the recommended print here.
