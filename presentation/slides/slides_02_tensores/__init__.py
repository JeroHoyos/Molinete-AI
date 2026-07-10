from .slide_que_es_un_tensor import SlideQueEsUnTensor
from .slide_softmax import SlideSoftmax
from .slide_strides import SlideStrides
from .slide_matmul import SlideMatmul
from .slide_simd import SlideSimd
from .slide_cache_blocking import SlideCacheBlocking
from .slide_parallel import SlideParallel
from .slide_batched_matmul import SlideBatchedMatmul


class SlidesTensores(SlideQueEsUnTensor, SlideSoftmax, SlideStrides, SlideMatmul, SlideSimd, SlideCacheBlocking, SlideParallel, SlideBatchedMatmul):
    pass
