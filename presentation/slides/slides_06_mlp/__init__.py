from .slide_layer_normalization import SlideLayerNormalization
from .slide_arquitectura_neurona import SlideArquitecturaNeurona
from .slide_zoom_neurona import SlideZoomNeurona
from .slide_activacion import SlideActivacion
from .slide_capa_transformer import SlideCapaTransformer
from .slide_residual import SlideResidual


class SlidesMLP(SlideLayerNormalization, SlideArquitecturaNeurona, SlideZoomNeurona, SlideActivacion, SlideCapaTransformer, SlideResidual):
    pass
