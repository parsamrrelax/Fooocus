import cv2
import numpy as np


def mask_to_rgba(mask):
    """Convert a 2D binary mask (0/255) to RGBA with white mask and transparent background."""
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    H, W = mask.shape[:2]
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    masked = mask > 127
    rgba[masked] = [255, 255, 255, 255]
    return rgba


def mask_to_data_url(mask):
    """Convert binary mask (0/255) to base64 PNG data URL."""
    if mask is None:
        return ''
    rgba = mask_to_rgba(mask)
    if rgba is None:
        return ''
    import io, base64
    from PIL import Image
    pil_img = Image.fromarray(rgba, mode='RGBA')
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{b64}'


def extract_2d_mask(mask):
    """Normalize any mask representation to a 2D uint8 array."""
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        if mask.ndim == 3:
            if mask.shape[2] == 4:
                # RGBA: take alpha channel if present and valid
                alpha = mask[:, :, 3]
                if np.any(alpha > 0):
                    return (alpha > 127).astype(np.uint8) * 255
            return (mask[:, :, 0] > 127).astype(np.uint8) * 255
        elif mask.ndim == 2:
            return (mask > 127).astype(np.uint8) * 255
    return None


def calculate_chroma_weighted_distance(lab_img, target_centers):
    """
    Compute minimum distance of every pixel in lab_img to the target cluster centers.
    Chroma (A, B) is weighted higher than Lightness (L) to be robust against fabric folds and shadows.
    """
    min_dist = np.full(lab_img.shape[:2], 1e6, dtype=np.float32)
    L = lab_img[:, :, 0]
    A = lab_img[:, :, 1]
    B = lab_img[:, :, 2]

    for center in target_centers:
        cL, cA, cB = center[0], center[1], center[2]
        dL = (L - cL) * 0.35
        dA = (A - cA) * 1.0
        dB = (B - cB) * 1.0
        dist = np.sqrt(dL ** 2 + dA ** 2 + dB ** 2)
        min_dist = np.minimum(min_dist, dist)

    return min_dist


def smart_edge_snap(image_rgb, stroke_mask, tolerance=50, expand=True, use_eraser=False, existing_mask=None):
    """
    Edge-aware snapping brush.
    Takes the user's rough brush stroke and snaps it to the underlying object's edges using
    chroma-weighted color modeling and graph-cut energy minimization (GrabCut).

    Parameters:
        image_rgb: (H, W, 3) numpy array uint8 RGB image
        stroke_mask: (H, W) or (H, W, 3/4) numpy array uint8
        tolerance: int 1-100 (controls sensitivity to color differences)
        expand: bool, whether to expand stroke to fill the object boundaries
        use_eraser: bool, if True, subtracts the snapped object from existing_mask
        existing_mask: optional (H, W) numpy array of previous mask state
    """
    if image_rgb is None or stroke_mask is None:
        return stroke_mask

    stroke_2d = extract_2d_mask(stroke_mask)
    if stroke_2d is None:
        return stroke_mask

    stroke_indices = np.where(stroke_2d > 127)
    if len(stroke_indices[0]) < 10:
        return stroke_2d

    H, W = image_rgb.shape[:2]
    if stroke_2d.shape[:2] != (H, W):
        stroke_2d = cv2.resize(stroke_2d, (W, H), interpolation=cv2.INTER_NEAREST)
        stroke_indices = np.where(stroke_2d > 127)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    y_min, y_max = int(np.min(stroke_indices[0])), int(np.max(stroke_indices[0]))
    x_min, x_max = int(np.min(stroke_indices[1])), int(np.max(stroke_indices[1]))

    stroke_h = y_max - y_min
    stroke_w = x_max - x_min
    margin = int(max(stroke_h, stroke_w) * 0.35 + 35) if expand else 15

    roi_y1 = max(0, y_min - margin)
    roi_y2 = min(H, y_max + margin + 1)
    roi_x1 = max(0, x_min - margin)
    roi_x2 = min(W, x_max + margin + 1)

    roi_img = image_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_stroke = stroke_2d[roi_y1:roi_y2, roi_x1:roi_x2] > 127

    rH, rW = roi_img.shape[:2]
    if rH < 8 or rW < 8:
        return stroke_2d

    # Scale down ROI if large for near-instant GrabCut graph cut
    max_dim = max(rH, rW)
    scale = 1.0
    if max_dim > 320:
        scale = 320.0 / max_dim
        small_w = max(16, int(rW * scale))
        small_h = max(16, int(rH * scale))
        small_img = cv2.resize(roi_img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small_stroke = cv2.resize(roi_stroke.astype(np.uint8), (small_w, small_h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        small_img = roi_img
        small_stroke = roi_stroke

    small_lab = cv2.cvtColor(small_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    s_idx = np.where(small_stroke)
    if len(s_idx[0]) < 5:
        return stroke_2d

    sampled_lab = small_lab[s_idx]

    # Outlier rejection on stroke pixels:
    # Most stroke pixels represent the intended object.
    # Outliers (accidental brush spill over background/skin) are rejected.
    median_color = np.median(sampled_lab, axis=0)
    d_median = np.sqrt(
        0.35 * (sampled_lab[:, 0] - median_color[0]) ** 2
        + (sampled_lab[:, 1] - median_color[1]) ** 2
        + (sampled_lab[:, 2] - median_color[2]) ** 2
    )

    base_dist = 12.0 + (float(tolerance) / 100.0) * 38.0
    inliers = d_median <= (base_dist * 1.3)
    
    if np.sum(inliers) >= 10:
        clean_sampled_lab = sampled_lab[inliers]
    else:
        clean_sampled_lab = sampled_lab

    # Cluster inlier stroke colors (to handle object highlights + shadows)
    if len(clean_sampled_lab) >= 50:
        k = 2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, _, centers = cv2.kmeans(clean_sampled_lab, k, None, criteria, 4, cv2.KMEANS_RANDOM_CENTERS)
        except Exception:
            centers = np.array([np.median(clean_sampled_lab, axis=0)], dtype=np.float32)
    else:
        centers = np.array([np.median(clean_sampled_lab, axis=0)], dtype=np.float32)

    dist_map = calculate_chroma_weighted_distance(small_lab, centers)

    # Initialize GrabCut mask
    # 0 = GC_BGD, 1 = GC_FGD, 2 = GC_PR_BGD, 3 = GC_PR_FGD
    gc_mask = np.full(small_stroke.shape, cv2.GC_BGD, dtype=np.uint8)
    gc_mask[dist_map > base_dist * 1.5] = cv2.GC_BGD
    gc_mask[(dist_map <= base_dist * 1.5) & (dist_map > base_dist * 0.85)] = cv2.GC_PR_BGD

    if expand:
        kernel_sz = max(7, min(small_img.shape[:2]) // 6)
        dilated = cv2.dilate(small_stroke.astype(np.uint8), np.ones((kernel_sz, kernel_sz), np.uint8)) > 0
        gc_mask[dilated & (dist_map <= base_dist)] = cv2.GC_PR_FGD
    else:
        gc_mask[small_stroke & (dist_map <= base_dist * 1.1)] = cv2.GC_PR_FGD

    # Inner core of inlier stroke is definite foreground
    inlier_mask = np.zeros(small_stroke.shape, dtype=bool)
    inlier_coords = (s_idx[0][inliers], s_idx[1][inliers])
    inlier_mask[inlier_coords] = True

    eroded = cv2.erode(inlier_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if np.sum(eroded) == 0:
        eroded = inlier_mask
    gc_mask[eroded & (dist_map <= base_dist * 0.75)] = cv2.GC_FGD

    # Accidental spillover outside target color is explicitly marked background!
    gc_mask[small_stroke & (dist_map > base_dist * 1.3)] = cv2.GC_BGD

    # Outer border of ROI is definite background
    gc_mask[0, :] = cv2.GC_BGD
    gc_mask[-1, :] = cv2.GC_BGD
    gc_mask[:, 0] = cv2.GC_BGD
    gc_mask[:, -1] = cv2.GC_BGD

    if not (np.any(gc_mask == cv2.GC_FGD) or np.any(gc_mask == cv2.GC_PR_FGD)):
        gc_mask[small_stroke & inlier_mask] = cv2.GC_FGD
    if not (np.any(gc_mask == cv2.GC_BGD) or np.any(gc_mask == cv2.GC_PR_BGD)):
        gc_mask[~small_stroke] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small_img, gc_mask, None, bgd_model, fgd_model, iterCount=4, mode=cv2.GC_INIT_WITH_MASK)
        small_res = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        small_res = (small_stroke & inlier_mask & (dist_map <= base_dist)).astype(np.uint8) * 255

    # Filter connected components: keep components touching the stroke inliers
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(small_res)
    cleaned_small = np.zeros_like(small_res)
    pts = np.where(inlier_mask & (dist_map <= base_dist))
    if len(pts[0]) > 0:
        seed_labels = np.unique(labels[pts])
        for lbl in seed_labels:
            if lbl != 0:
                cleaned_small[labels == lbl] = 255
    else:
        cleaned_small = small_res

    # Scale back to full ROI resolution
    if scale != 1.0:
        full_res = cv2.resize(cleaned_small, (rW, rH), interpolation=cv2.INTER_LINEAR)
        full_bin = (full_res > 127).astype(np.uint8) * 255
    else:
        full_bin = cleaned_small

    # Morphological clean up to smooth boundaries
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    full_bin = cv2.morphologyEx(full_bin, cv2.MORPH_CLOSE, kernel_smooth)

    snapped_roi_mask = np.zeros_like(stroke_2d)
    snapped_roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = full_bin

    if use_eraser:
        # Subtract the snapped object from existing mask
        if existing_mask is not None:
            base_mask = extract_2d_mask(existing_mask)
            if base_mask.shape[:2] != (H, W):
                base_mask = cv2.resize(base_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            base_mask = stroke_2d.copy()
        base_mask[snapped_roi_mask > 127] = 0
        return base_mask
    else:
        # If existing mask is given, union with it
        if existing_mask is not None:
            base_mask = extract_2d_mask(existing_mask)
            if base_mask.shape[:2] != (H, W):
                base_mask = cv2.resize(base_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            return np.maximum(base_mask, snapped_roi_mask)
        return snapped_roi_mask
