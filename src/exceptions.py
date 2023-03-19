class CalibrationError(Exception):
    def __init__(self, message):
        self.message = message

class IntrinsicParametersNotFoundError(Exception):
    def __init__(self, message):
        self.message = message