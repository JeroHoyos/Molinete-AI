from .slide_mha_acto1_intuicion import SlideMhaActo1Intuicion
from .slide_mha_acto2_qkv import SlideMhaActo2Qkv
from .slide_mha_acto3_formula_y_flujo import SlideMhaActo3FormulaYFlujo
from .slide_mha_acto4_multihead import SlideMhaActo4Multihead


class SlidesAtencion(SlideMhaActo1Intuicion, SlideMhaActo2Qkv, SlideMhaActo3FormulaYFlujo, SlideMhaActo4Multihead):
    pass
