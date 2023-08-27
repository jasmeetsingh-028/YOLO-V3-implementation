import torch 
from iou import intersection_over_union

def nms(bboxes, 
        iou_th,
        probablity_th,
        box_format = "corners",):
    #prediction = [class [[1 - car, probability of that bounding box 0.9, x1,y1, x2, y2],[],[]]

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > probablity_th]  #checking the probablity thereshold x[1]
    bboxes_after_nms = []
    bboxes = sorted(bboxes, key = lambda x: x[1], reverse = True)  #sorted bouding boxes with highest probablity x[1] at the begnning
    
    while bboxes:
        chosen_box = bboxes.pop(0)  #highest probability box

        bboxes = [

            box for box in bboxes
            if box[0] != chosen_box[0]  #not of the same class
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),   #checking if iou is less than the threshold
                box_format = box_format,

            ) < iou_th
        ]

        bboxes_after_nms.append(chosen_box)

