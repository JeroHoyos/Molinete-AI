from ._helpers import HelpersEmbeddings
from .slide_embeddings import SlideEmbeddings
from .slide_position_embeddings import SlidePositionEmbeddings


class SlidesEmbeddings(HelpersEmbeddings, SlideEmbeddings, SlidePositionEmbeddings):
    pass
