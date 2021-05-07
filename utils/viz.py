# -*- coding: utf-8 -*-
import cv2


def visualize_detections(image, boxes, preds):
    for (box, pred) in zip(boxes, preds):
        # unpack the bounding box
        x, y, width, height = box["box"]
        # determine class
        label = "Masque" if pred > 0.5 else "Sans masque"
        color = (0, 255, 0) if label == "Masque" else (255, 0, 0)
        # include the probability in the label
        cv2.rectangle(image, (x, y + height - 35), (x + width, y + height), color, cv2.FILLED)
        cv2.putText(image, label, (x + 6, y + height - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 1)
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
    return image


def visualize_mask(image, pred):
    label = "Masque" if pred > 0.5 else "Sans masque"
    color = (0, 255, 0) if label == "Masque" else (0, 0, 255)
    cv2.putText(image, label, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
    return image
