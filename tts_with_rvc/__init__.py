__all__ = ["TTS_RVC", "OnnxRVCConverter"]
__version__ = "0.1.10"


def __getattr__(name):
    if name == "TTS_RVC":
        from .inference_onnx import TTS_RVC

        return TTS_RVC
    if name == "OnnxRVCConverter":
        from .converter_onnx import OnnxRVCConverter

        return OnnxRVCConverter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
