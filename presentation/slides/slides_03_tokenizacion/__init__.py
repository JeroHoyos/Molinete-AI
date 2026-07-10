from .slide_problema_strawberry import SlideProblemaStrawberry
from .slide_byte_pair_encoding import SlideBytePairEncoding
from .slide_tokenizacion import SlideTokenizacion


class SlidesTokenizacion(SlideProblemaStrawberry, SlideBytePairEncoding, SlideTokenizacion):
    pass
