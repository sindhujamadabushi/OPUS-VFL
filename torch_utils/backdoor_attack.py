def find_best_patch_location(saliency_map, patch_size=5):
    """
    Finds the (row, col) in `saliency_map` where a patch of size patch_size x patch_size
    has the highest average saliency. This is a naive, sliding-window approach.
    """
    H, W = saliency_map.shape
    best_avg = -1
    best_rc = (0, 0)
    for r in range(H - patch_size + 1):
        for c in range(W - patch_size + 1):
            window = saliency_map[r:r+patch_size, c:c+patch_size]
            avg_sal = window.mean()
            if avg_sal > best_avg:
                best_avg = avg_sal
                best_rc = (r, c)
    return best_rc

def insert_trigger_patch(image_tensor, row, col, patch_size=5, trigger_value=1.0):
    """
    Inserts a simple patch (square) of `trigger_value` into `image_tensor`
    at location (row, col). For color images, we do this on each channel.
    Expects image_tensor shape: (3, H, W) or (C, H, W).
    """
    c, h, w = image_tensor.shape
    # clamp if patch extends outside
    end_r = min(row + patch_size, h)
    end_c = min(col + patch_size, w)
    image_tensor[:, row:end_r, col:end_c] = trigger_value
    return image_tensor
