import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manim import *
from manim_slides import Slide
from manim_code_blocks import *
import os
from colores import *
from objetos import *
from slides import Slides
from base import SlideBase


class Presentacion(Slides,
                   SlideBase,
                   Slide):

    def construct(self):
        self._slide_actual = 0
        self._TOTAL_SLIDES = self._contar_total_slides()
        self.camera.background_color = WHITE

        self.slide_pronto_iniciamos()
        # self.slide_introduction()
        # self.slide_credits()
        # self.slide_que_es_transformer()
        # self.slide_por_que_rust()
        # self.slide_rust_python_bridge()
        # self.slide_molinete_ai()

        # self.slide_roadmap()

        # self.slide_que_es_un_tensor()
        # self.slide_strides()
        self.mostrar_snippet("tensor.rs")

        # self.slide_matmul()
        self.mostrar_snippet("matmul_base.rs")

        # self.slide_simd()
        self.mostrar_snippet("simd_vectorization.rs")

        # self.slide_cache_blocking()
        self.mostrar_snippet("cache_blocking.rs")

        # self.slide_parallel()
        self.mostrar_snippet("parallel.rs")

        # self.slide_problema_strawberry()
        # self.slide_tokenizacion()
        self.mostrar_snippet("BDPtokenizer.rs")

        # self.slide_byte_pair_encoding()
        self.mostrar_snippet("pair_counts.rs")

        # self.slide_embeddings()
        # self.slide_position_embeddings()
        self.mostrar_snippet("embedding.rs")

        # self.slide_mha_acto1_intuicion()
        # self.slide_mha_acto2_qkv()
        # self.slide_mha_acto3_formula_y_flujo()
        # self.slide_mha_acto4_multihead()
        self.mostrar_snippet("attention.rs")

        # self.slide_arquitectura_neurona()
        # self.slide_zoom_neurona()
        self.mostrar_snippet("mlp_forward.rs")
        # self.slide_activacion()
        self.mostrar_snippet("gelu.rs")

        # self.slide_layer_normalization()
        self.mostrar_snippet("normalization.rs")
        # self.slide_residual()
        # self.slide_capa_transformer()
        self.mostrar_snippet("block_backward.rs")

        # self.slide_softmax()
        self.mostrar_snippet("softmax.rs")

        # self.slide_temperature()
        self.mostrar_snippet("temperature.rs")

        # self.slide_entrenamiento()
        # self.slide_training_metrics()
        self.mostrar_snippet("compute_loss.rs")

        # self.slide_descenso_gradiente()
        # self.slide_backpropagation()
        self.mostrar_snippet("linear_backward.rs")

        # self.slide_adam()
        self.mostrar_snippet("adamw_update.rs")

        # self.slide_dropout()
        self.mostrar_snippet("dropout.rs")

        # self.slide_conteo_parametros()

        # self.slide_model_in_action()
        # self.slide_final()

