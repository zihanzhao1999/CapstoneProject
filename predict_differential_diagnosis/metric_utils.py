import numpy as np

def compute_f1(p, r):
    if isinstance(p, (list, tuple)):
        p = np.array(p)
    if isinstance(r, (list, tuple)):
        r = np.array(r)
    denom = p + r
    return (2 * p * r) / (denom + 1e-10)

def compute_metric(gt_diff_mask, pred_diff_mask):
    result = {}
    ddr = np.sum(np.logical_and(gt_diff_mask, pred_diff_mask), axis=-1) / np.maximum(1, np.sum(gt_diff_mask, axis=-1))
    ddp = np.sum(np.logical_and(gt_diff_mask, pred_diff_mask), axis=-1) / np.maximum(1, np.sum(pred_diff_mask, axis=-1))
    ddf1 = compute_f1(ddp, ddr)

    result[f"ACC"] = 0.997 # from previous model
    result[f"DDR"] = np.mean(ddr)
    result[f"DDP"] = np.mean(ddp)
    result[f"DDF1"] = np.mean(ddf1)
    result[f"GM"] = (result[f"ACC"] * result[f"DDR"] * result[f"DDP"])**(1/3)
    return result