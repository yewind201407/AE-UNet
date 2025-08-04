import numpy as np

class SegmentationMetrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        self.tp = np.zeros(self.n_classes)
        self.fp = np.zeros(self.n_classes)
        self.fn = np.zeros(self.n_classes)

    def update(self, preds, labels):
        preds = preds.flatten()
        labels = labels.flatten()

        for i in range(self.n_classes):
            self.tp[i] += np.sum((preds == i) & (labels == i))
            self.fp[i] += np.sum((preds == i) & (labels != i))
            self.fn[i] += np.sum((preds != i) & (labels == i))

    def get_scores(self):
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-10)
        precision = self.tp / (self.tp + self.fp + 1e-10)
        recall = self.tp / (self.tp + self.fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # For binary case, report scores for class 1
        scores = {
            'mIoU': np.mean(iou),
            'Precision': precision[1] if self.n_classes > 1 else precision[0],
            'Recall': recall[1] if self.n_classes > 1 else recall[0],
            'F1': f1[1] if self.n_classes > 1 else f1[0]
        }
        return scores