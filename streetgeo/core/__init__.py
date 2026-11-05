"""S.V.G.T. application package."""


class EmptyPredictionDomainError(ValueError):
    """No model-generated coordinate satisfies the selected prediction domain."""


class UnsupportedViewCountError(ValueError):
    def __init__(self, maximum):
        self.maximum = maximum
        super().__init__(f"provide 1..{maximum} images of the SAME location")
