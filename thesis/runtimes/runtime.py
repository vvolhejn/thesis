TEMP_DIR = "/tmp"


class Runtime:
    def __init__(self):
        self.convert_called = False

    def convert(self, orig_model, get_batch_fn=None):
        self.convert_called = True
        self.orig_model = orig_model

    def run(self, data):
        assert self.convert_called, "No model was converted, call convert() first."

    def run_timed(self, data, timer):
        with timer:
            return self.run(data)

    def get_name(self):
        raise NotImplementedError

    def __repr__(self):
        return self.get_name()

    def get_id(self):
        return self.get_name() + "_" + hex(id(self))


class NeedsPyTorchModel:
    """
    Used only to tag runtimes that converts a PyTorch model rather than a Keras one
    """

    pass
