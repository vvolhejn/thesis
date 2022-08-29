# Accelerating Neural Audio Synthesis

This is the code accompanying my Master's thesis at ETH Zürich titled _Accelerating Neural Audio Synthesis_. The goal was to create a fast model for synthesizing musical audio in real time, based on [DDSP](https://arxiv.org/abs/2001.04643) and [RAVE](https://arxiv.org/abs/2111.05011). The resulting DDSP-based model, DDSP-CNN-Tiny, has less than 2500 parameters and runs at over 800x real-time on a CPU, while maintaining the quality of the original DDSP-full with 6M parameters.

Audio examples for the different models are available [here](https://vvolhejn.github.io/thesis/).

This is research code that needs modifying to be reusable (there are some hardcoded file paths, for instance), and active maintenance is not planned. Nevertheless, parts of the code can be useful for others:

- The code to define the DDSP-CNN-Tiny model is in [ddspae-cnn.gin](gin/ddspae-cnn.gin) (but set `CustomDilatedConvDecoder.casual=True` and `CustomDilatedConvDecoder.ch=8`) and [dilated_conv.py](thesis/dilated_conv.py)
- [thesis/runtimes/](thesis/runtimes/) contains code to convert models from TensorFlow, PyTorch or ONNX to various deep learning runtime libraries: TFLite, TorchScript, ONNX Runtime, OpenVINO, TVM and DeepSparse. Where applicable, it also includes code for quantizing the models through static or dynamic quantization.
- [thesis/](thesis/) includes a TensorFlow+DDSP re-implementation of components of [NEWT](https://arxiv.org/abs/2107.05050) and [RAVE](https://arxiv.org/abs/2111.05011), two models that were originally in PyTorch. There are modules such as [PQMF analysis and synthesis](thesis/pqmf.py) and [learnable waveshapers](thesis/newt.py).
- The code depends on a [modified fork of the DDSP library](https://github.com/vvolhejn/ddsp), which includes Weights and Biases integration and various other changes that couldn't be done without modifying the library.
- Notebooks to produce the figures seen in the thesis: [thesis-runtimes-plots.ipynb](notebooks/thesis-runtimes-plots.ipynb), [thesis-experiments3-plots.ipynb](notebooks/thesis-experiments3-plots.ipynb) and [survey-evaluation.ipynb](notebooks/survey-evaluation.ipynb).
- [A notebook](notebooks/download-samples.ipynb) to prepare the audio examples for GitHub pages.