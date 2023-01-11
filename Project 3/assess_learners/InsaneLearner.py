import numpy as np
import LinRegLearner as lrl
import BagLearner as bl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, bags=20) for i in range(0, 20)]
    def author(self):
        return "mworthley3"
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)
    def query(self, points):
        results = np.zeros((points.shape[0], ))
        for learner in self.learners:
            results += learner.query(points)
        return results / len(self.learners)