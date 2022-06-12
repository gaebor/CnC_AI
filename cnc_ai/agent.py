class AbstractAgent:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def learn(self, inputs, outputs, rewards):
        pass

    def save(self, path):
        pass

    @staticmethod
    def load(path):
        pass
