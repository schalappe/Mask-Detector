"""
    Script used to apply some visualization
"""
# -*- coding: utf-8 -*-
import cv2
from numpy import ndarray


def visualize_detections(image: ndarray, boxes: list, preds: list) -> ndarray:
    """
    Add color box on the face. Green if mask Red else
    Args:
        image (ndarray): Image to process
        boxes (list): List of coordinates
        preds (list): List of prediction probability

    Returns:
        ndarray: Image processed
    """
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


def visualize_mask(image: ndarray, pred: float) -> ndarray:
    """
    Add label to an image
    Args:
        image (ndarray): Image to process
        pred (float): Prediction probability

    Returns:
        ndarray: Image processed
    """
    label = "Masque" if pred > 0.5 else "Sans masque"
    color = (0, 255, 0) if label == "Masque" else (0, 0, 255)
    cv2.putText(image, label, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
    return image
