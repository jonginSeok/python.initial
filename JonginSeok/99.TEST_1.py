import json
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def evaluate(pred_path, gt_path, iou_thresh=0.5):
    with open(pred_path) as f:
        preds = json.load(f)
    with open(gt_path) as f:
        gts = json.load(f)

    # Group by image and class
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)

    for ann in gts:
        gt_by_image[ann['image_id']].append(ann)
    for ann in preds:
        pred_by_image[ann['image_id']].append(ann)

    # Metrics per class
    class_metrics = defaultdict(lambda: {
        "TP": 0, "FP": 0, "FN": 0, "instances": 0
    })

    for image_id in gt_by_image:
        gt_annots = gt_by_image[image_id]
        pred_annots = pred_by_image.get(image_id, [])

        matched = set()
        for pred in pred_annots:
            best_iou = 0
            best_gt_idx = -1
            for idx, gt in enumerate(gt_annots):
                if gt['category_id'] != pred['category_id'] or idx in matched:
                    continue
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            cls = pred['category_id']
            if best_iou >= iou_thresh:
                class_metrics[cls]["TP"] += 1
                matched.add(best_gt_idx)
            else:
                class_metrics[cls]["FP"] += 1

        # FN: ground truths not matched
        for idx, gt in enumerate(gt_annots):
            cls = gt['category_id']
            class_metrics[cls]["instances"] += 1
            if idx not in matched:
                class_metrics[cls]["FN"] += 1

    # Final metrics
    results = {}
    for cls, m in class_metrics.items():
        TP, FP, FN = m["TP"], m["FP"], m["FN"]
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        results[cls] = {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "Instances": m["instances"]
        }

    return results



metrics = evaluate("predictions.json", "ground_truth.json", iou_thresh=0.5)
# random
for cls_id, stats in metrics.items():
    print(f"Class {cls_id}: {stats}")
# sort
# for cls_id in sorted(metrics.keys()):
#     stats = metrics[cls_id]
#     print(f"Class {cls_id}: {stats}")