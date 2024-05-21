from abc import ABC, abstractmethod
import numpy as np
from utils import compute_unweighted_accuracy, compute_weighted_f1


class BaseModel(ABC):

    def __init__(
        self,
        model,
        trained: bool = False
    ) -> None:
        self.model = model
        self.trained = trained

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def predict(self, samples: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: str, name: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, name: str):
        pass

    @classmethod
    @abstractmethod
    def make(cls):
        pass

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray, num_classes: int) -> None:
        predictions = self.predict(x_test)
        true_labels = y_test

        # unweighted accuracy
        unweightet_correct = [0] * num_classes
        unweightet_total = [0] * num_classes

        # weighted f1
        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes

        total = len(true_labels)
        correct = (predictions == true_labels).sum()

        for i in range(len(true_labels)):
            unweightet_total[true_labels[i]] += 1
            if predictions[i] == true_labels[i]:
                unweightet_correct[true_labels[i]] += 1
                tp[true_labels[i]] += 1
            else:
                fp[predictions[i]] += 1
                fn[true_labels[i]] += 1

        weighted_acc = correct / total
        unweighted_acc = compute_unweighted_accuracy(unweightet_correct, unweightet_total)
        weighted_f1 = compute_weighted_f1(tp, fp, fn, unweightet_total)

        return (
            weighted_acc,
            unweighted_acc,
            weighted_f1,
        )
