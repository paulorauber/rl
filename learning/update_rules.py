class SampleAverageUpdate:

    def __call__(self, q, reward, k):
        return q + (1. / k) * (reward - q)


class ConstantStepUpdate:

    def __init__(self, step):
        self.step = step

    def __call__(self, q, reward, k=None):
        return q + self.step * (reward - q)
