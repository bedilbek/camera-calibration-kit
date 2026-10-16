# Calibration Review Metrics: Calculation and Meaning

This guide explains the four columns in **Refit + compare models**. It uses short sentences and consistent terms. It follows the writing principles of [ASD-STE100](https://www.asd-ste100.org/STE_faq.html). It has not had a formal STE compliance check.

## Terms and common calculation

An **image** shows a ChArUco board. A **corner** is one detected board point. A **model** contains the camera matrix $K$ and distortion values $d$. Each image also has a board pose $(R_i,t_i)$. A **pixel** is one image coordinate unit.

For corner $j$ in image $i$, let $\mathbf{u}_{ij}$ be its detected pixel position. The model projects its known board position $\mathbf{X}_{ij}$ to $\hat{\mathbf{u}}_{ij}$:

$$
\hat{\mathbf{u}}_{ij}=\operatorname{project}(\mathbf{X}_{ij};K,d,R_i,t_i).
$$

$$
e_{ij}=\left\|\mathbf{u}_{ij}-\hat{\mathbf{u}}_{ij}\right\|_2.
$$

The value $e_{ij}$ is the distance between the detected and projected corners, in pixels. OpenCV estimates $K$, $d$, and the poses during a fit.

## Fit RMS px

The app fits the model to all selected images. It then reports OpenCV's root mean square (RMS) reprojection error:

$$
\mathrm{Fit\ RMS}=\sqrt{\frac{1}{N}\sum_i\sum_j e_{ij}^{2}}.
$$

Here, $N$ is the total number of detected corners in the selected images. This is a corner-weighted score. It is not the mean of the per-image RMS values. A smaller value means a closer fit to these same images. It does not prove accuracy on new images.

## Held-out RMS px

The app sorts selected image names. It puts alternate images into sets A and B. It fits one model to A and a second model to B. Each set must contain at least ten detected images.

For each image in B, the app keeps A's $K$ and $d$. It estimates a new board pose from **all corners in that B image**. It then computes reprojection distances for those same corners. It repeats this process with B's model on images in A.

$$
\mathrm{Held\text{-}out\ RMS}
=\sqrt{\frac{1}{N_H}\sum_{i,j\,\in\,H} e_{ij}^{2}}.
$$

Here, $H$ contains the evaluated images from both directions. $N_H$ is their total evaluated corner count. The camera parameters come from other images. The pose comes from the test image itself. Thus, this score tests camera-model transfer, but it is not an independent test of each corner position. The two sets also come from one capture session.

## Split ray median px and max px

The app uses the A and B models again. It places a 9 × 5 grid across the full image, including its edges. For each grid pixel $\mathbf{p}_g$, it reverses distortion with each model. It expresses both results in the same reference pixel plane:

$$
\mathbf{q}_{A,g}=\operatorname{undistort}(\mathbf{p}_g;K_A,d_A,K_{\mathrm{ref}}).
$$

$$
\mathbf{q}_{B,g}=\operatorname{undistort}(\mathbf{p}_g;K_B,d_B,K_{\mathrm{ref}}).
$$

$$
\Delta_g=\|\mathbf{q}_{A,g}-\mathbf{q}_{B,g}\|_2.
$$

The **median** is the middle value of the 45 distances $\Delta_g$. The **max** is the largest distance. $K_{\mathrm{ref}}$ comes from the original `intrinsics.json`. It sets the pixel scale for this comparison.

These values measure **disagreement between models** in a common pixel plane. They are not corner errors, angle errors, or distances on the board. A large edge value can show that the image set gives weak support near an edge. Neither model is compared with a true ray direction.

## What split-fit stability means

**Split-fit stability** is the repeatability of a camera model when different images are used for the fit. The app fits set A and set B separately, then compares their results. It does **not** calculate a fifth metric called “split-fit stability.”

The **split-ray median** and **split-ray max** are the app's direct measures of this stability. Smaller values mean that the two fits produce more similar ray maps. The held-out RMS is a related transfer check. It measures corner reprojection after the app estimates a new board pose. A low held-out RMS does not guarantee stable rays across the full frame.

The app warns when the split-ray median exceeds 5 px. This is a prompt to capture better views, not a universal pass limit. Two similar fits can share the same error. Use new images and a physical accuracy check when accuracy matters.

## How to use the four values

Compare rows for the same selected images. The rows use four, five, or eight active OpenCV distortion coefficients. A more flexible model can lower Fit RMS and still give unstable rays. Inspect coverage, residual plots, and new images before you accept a model. Do not compare a split-ray value directly with an RMS value: they measure different quantities, although both use pixels.

For example, the 31-image session `20261014-064430` gave **0.612 px Fit RMS** and **35.1 px split-ray median** with five coefficients. The first value describes fitted corners. The second describes a difference between two inferred ray maps. These values do not contradict each other.

The calculations are in [`review.py`](../src/calibration_demo/review.py). The table is in [`review_gui.py`](../src/calibration_demo/review_gui.py).
