import torch
import torch.nn as nn

class F1ScoreLoss(nn.Module):
    def __init__(self, num_classes, epsilon=1e-7, average='macro', weight=None):
        """
        F1 Score Loss function for PyTorch with multi-class support and class weighting.
        
        Args:
            num_classes (int): Number of classes.
            epsilon (float): A small constant to prevent division by zero.
            average (str): Determines the type of averaging performed. 
                           'macro' calculates F1 independently for each class and averages them.
                           'weighted' averages F1 scores weighted by support (number of true instances per class).
            class_weights (torch.Tensor, optional): Weights for each class (num_classes,).
        """
        super(F1ScoreLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.average = average
        self.class_weights = weight

    def forward(self, y_pred, y_true):
        """
        Computes the F1 Score loss for multi-class classification.
        
        Args:
            y_pred (torch.Tensor): Predicted logits (batch_size, num_classes).
            y_true (torch.Tensor): Ground truth labels (batch_size,) with class indices.
            
        Returns:
            torch.Tensor: F1 Score loss value.
        """
        y_pred = torch.softmax(y_pred, dim=1)  # Convert logits to probabilities
        y_true = y_true.to(torch.long)  # Ensure y_true is of type long

        # Compute true positives, false positives, and false negatives for each class
        true_positives = (y_pred * y_true).sum(dim=0)
        false_positives = ((1 - y_true) * y_pred).sum(dim=0)
        false_negatives = (y_true * (1 - y_pred)).sum(dim=0)

        # Precision and recall for each class
        precision = true_positives / (true_positives + false_positives + self.epsilon)
        recall = true_positives / (true_positives + false_negatives + self.epsilon)

        # F1 score for each class
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)

        # Handle class weights if provided
        if self.class_weights is not None:
            class_weights = self.class_weights.to(y_pred.device)
            f1 *= class_weights

        # Handle averaging
        if self.average == 'macro':
            f1_loss = 1 - f1.mean()  # Macro-average
        elif self.average == 'weighted':
            support = y_true.sum(dim=0)  # Number of true instances per class
            f1_loss = 1 - (f1 * support).sum() / support.sum()  # Weighted-average
        else:
            raise ValueError("Average must be either 'macro' or 'weighted'")

        return f1_loss
