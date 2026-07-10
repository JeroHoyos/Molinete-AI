from .slide_rust_python_bridge import SlideRustPythonBridge
from .slide_model_in_action import SlideModelInAction
from .slide_final import SlideFinal


class SlidesFinal(SlideRustPythonBridge, SlideModelInAction, SlideFinal):
    pass
