from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


class Validator():

    def __init__(self, y_truth, y_pred):
        self.truth = y_truth
        self.pred = y_pred
        self.acc = None
        self.report = None
        self.get_stats()

    def get_stats(self):
        self.acc = accuracy_score(self.truth, self.pred)
        self.report = classification_report(self.truth, self.pred)

    @property
    def x(self):
        return self._x