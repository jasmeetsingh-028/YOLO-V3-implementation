import torch
from collections import Counter
from iou import intersection_over_union

def mean_average_precision(
        pred_boxes,     #all training boxes over all training examples
        true_boxes,
        iou_threshold = 0.5,
        box_format = 'corners',
        num_classes = 20
):
    #pred_boxes (list) = [[train_idx, class_pred, prob, x1, y1, x2, y2]...]  train_idx - img that this bounding box comes from
    average_precisions = []   #we will average precisions for each class
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:   #adding preditions and true labels for one class
            if detection[1] == c:
                detection.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        #img 0 has 3 bboxes
        #img 1 has 5 boxes
        #amount_boxes = {0:3,1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        
        #amount_bboxes = {0: ([0,0,0]), 1: ([0,0,0,0,0])}
        
        detections.sort(key = lambda x: x[2], reverse = True)  #descending sort over probabilities

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in detections:
            
