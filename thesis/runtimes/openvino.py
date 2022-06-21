from . import ONNXRuntime


class OpenVINO(ONNXRuntime):
    def __init__(self, *args, **kwargs):
        # import openvino.runtime

        super().__init__(*args, **kwargs)

    def convert(self, orig_model, get_batch_fn=None):
        # from openvino.inference_engine import IECore
        # self.ie = IECore()

        from openvino.runtime import Core

        self.ie = Core()
        self.ie.set_property(
            "CPU", {"CPU_THREADS_NUM": str(util.get_n_cpus_available())}
        )
        print("OPENVINO THREADS", self.ie.get_property("CPU", "CPU_THREADS_NUM"))

        super().convert(orig_model, get_batch_fn)
        self.save_dir = os.path.join(TEMP_DIR, self.get_id())
        os.makedirs(self.save_dir)

        mo_command = [
            "mo",
            "--input_model",
            # self.optimized_model_path,
            self.save_path,
            "--output_dir",
            self.save_dir,
        ]
        subprocess.run(mo_command)

        model = self.ie.read_model(
            model=os.path.join(
                self.save_dir,
                os.path.splitext(os.path.basename(self.save_path))[0] + ".xml",
            )
        )
        self.compiled_model = self.ie.compile_model(model=model, device_name="CPU")
        self.input_names = [x.any_name for x in self.compiled_model.inputs]
        # self.request = self.compiled_model.create_infer_request()

        # model = self.ie.read_network(
        #     model=os.path.join(
        #         self.save_dir,
        #         os.path.splitext(os.path.basename(self.save_path))[0] + ".xml",
        #     )
        # )
        #
        # self.compiled_model = self.ie.load_network(network=model, device_name="CPU")
        # self.input_names = list(self.compiled_model.input_info)
        # self.request = self.compiled_model.create_infer_request()

    def run(self, data):
        # Checks that the model has been converted
        Runtime.run(self, data)

        # output = self.compiled_model.infer({self.input_names[0]: data})
        # output = self.compiled_model({self.input_names[0]: data})
        output = self.compiled_model(inputs=[data])
        # output = self.request.infer({self.input_names[0]: data})

        output_list = list(output.values())
        assert len(output_list) == 1, "Only one output was expected."
        return output_list[0]
