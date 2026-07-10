from .slide_entrenamiento import SlideEntrenamiento
from .slide_descenso_gradiente import SlideDescensoGradiente
from .slide_backpropagation import SlideBackpropagation
from .slide_adam import SlideAdam
from .slide_dropout import SlideDropout
from .slide_training_metrics import SlideTrainingMetrics
from .slide_temperature import SlideTemperature


class SlidesEntrenamiento(SlideEntrenamiento, SlideDescensoGradiente, SlideBackpropagation, SlideAdam, SlideDropout, SlideTrainingMetrics, SlideTemperature):
    pass
