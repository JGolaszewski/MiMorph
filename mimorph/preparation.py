import cv2
import numpy as np

def tissue_create_mask(tissue):
    gray = tissue if tissue.ndim == 2 else np.max(tissue[:,:,:3], axis=2)
    mask = (gray < 200).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    floodfilled = closed_mask.copy()
    h, w = closed_mask.shape
    mask_floodfill = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(floodfilled, mask_floodfill, (0,0), 255)
    floodfilled_inv = cv2.bitwise_not(floodfilled)
    filled_mask = closed_mask | floodfilled_inv

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_mask = (labels == largest_label).astype(np.uint8)
    else:
        largest_mask = filled_mask

    return largest_mask.astype(bool)