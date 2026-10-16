# Camera Models and What Calibration Estimates

For a pinhole camera, a 3D point is first expressed relative to the camera and divided by its depth. The intrinsic matrix `K` then converts that normalized ray to a pixel: `fx` and `fy` are focal lengths in pixels, while `cx` and `cy` locate the principal point. Lens-distortion coefficients adjust where a ray appears, especially away from the image centre. Intrinsics describe **one camera at one image resolution and lens state**; matching webcams do not share a guaranteed calibration.

This app currently compares three OpenCV pinhole/Brown-family variants:

| Choice | Estimated distortion terms | Reason to consider it |
|---|---|---|
| 4 coefficients | `k1`, `k2` (radial); `p1`, `p2` (tangential); `k3 = 0` | Simpler fit when higher-order distortion is weak or poorly observed. |
| 5 coefficients | Adds radial `k3` | The app's original default. |
| 8 coefficients | Adds rational denominator terms `k4`–`k6` | More flexibility for lenses that need it, but more opportunity to overfit. |

The GUI does not yet fit fisheye, Double Sphere, or spline camera models. A lower training RMS after adding terms is **not** enough to select a model. Compare residual structure, held-out behavior, and the agreement of independently fitted viewing rays. Model parameters can shift substantially while reprojection RMS stays low if the board samples only a small part of the field or a narrow range of poses.

The board's square size defines **metric scale**. If a print is uniformly scaled but its actual size is entered incorrectly, intrinsic pixel parameters may still look plausible, while estimated board distance and later stereo baselines or 3D lengths can be wrong. Uneven printing or a warped mounting surface can also bias the lens fit. Measure the print in both directions and keep it flat.

For confidence, combine: (1) varied and well-distributed board images; (2) low, pattern-free residuals; (3) stable fits when images are split or repeated; and (4) an **independent physical check** such as a known-length target at relevant positions and distances. A low training RMS alone answers only whether the model fits its own observations.
