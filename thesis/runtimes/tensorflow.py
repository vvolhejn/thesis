from . import Runtime


class TensorFlow(Runtime):
    def run(self, data):
        super().run(data)

        output = self.orig_model(data)
        return output

    def get_name(self):
        return f"TensorFlow"
