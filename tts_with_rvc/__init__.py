__version__ = "0.1.10"

__all__ = [
    "TTS_RVC",
    "RVCConverter",
    "rvc_convert",
    "get_voices",
    "speech",
    "tts_communicate",
]


def __getattr__(name):
    if name == "TTS_RVC":
        from .inference import TTS_RVC

        return TTS_RVC
    if name in {"get_voices", "speech", "tts_communicate"}:
        from . import inference

        return getattr(inference, name)
    if name in {"RVCConverter", "rvc_convert"}:
        from . import vc_infer

        return getattr(vc_infer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
