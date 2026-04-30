from manim import *
from manim_slides import Slide
from manim_code_blocks import *
import numpy as np
import random
import math
import os

FUENTE = "Goudy Old Style"

MARRON_OSCURO      = "#3D3834"
NARANJA_TERRACOTA  = "#A36536"
PAPEL_CREMA        = "#F2E6D8"
PAPEL_TAN          = "#B78B68"
FONDO_CAJA         = "#FCF3E4"
CAJA_INFERIOR      = "#E0C2A8"
TINTA_NEGRA        = "#1A1A1A"
NEGRO_SUAVE        = "#1E1E1E"
BLANCO             = "#FFFFFF"
MADERA_OSCURA      = "#3E2723"
MADERA_CLARA       = "#5D4037"
LADRILLO           = "#8D6E63"
TERRACOTA          = "#BF360C"
TEJA               = "#D84315"
HIERRO             = "#424242"
ACERO              = "#78909C"
PERGAMINO          = "#F4E4BC"
ORO_VIEJO          = "#D4AF37"
LATON              = "#B5A642"
ROJO_SANGRE        = "#8B0000"
AZUL_NOCHE         = "#000080"
VERDE_BOSQUE       = "#228B22"
TIERRA_MANCHEGA    = "#D4B872"
ARENA_MANCHEGA     = "#E6D3A8"
BARRO_MANCHEGO     = "#8B5A2B"
ROJO_TOMATE        = "#E24A4A"
VERDE_OLIVA        = "#6B8E23"
LAVANDA            = "#C4C4FF"
SALMON_CLARO       = "#F2D5CE"
CREMA_CALIDA       = "#E8DCC4"
BEIGE_MEDIO        = "#D9C8AA"
LADRILLO_VIVO      = "#C0573E"
SALMON_ATENCION    = "#E6A87C"
ARENA_DORADA       = "#C2B280"
NARANJA_CLARO      = "#FFCC99"
AMARILLO_PALIDO    = "#FFFFCC"
MENTA_PALIDA       = "#CCFFCC"
PERGAMINO_CLARO    = "#F4EBD0"
ROJO_MAC           = "#FF5F56"
AMARILLO_MAC       = "#FFBD2E"
VERDE_MAC          = "#27C93F"
OCRE_CERVANTINO    = "#C9A84C"
MARRON_QUIJOTE     = "#8B2500"
ROJO_CONTRA        = "#B33A3A"
FONDO_ESCUDO = "#37474F"
METAL_CLARO = "#90A4AE"
METAL_OSCURO = "#546E7A"
MADERA_VIEJA = "#4E4039"
CUERO_GASTADO = "#2E1E18"
TELA_DESLAVADA = "#6D4C41"

BUFF_S   = 0.15
BUFF_M   = 0.25
BUFF_L   = 0.4

FONT_TITULO  = 35
FONT_CUERPO  = 24
FONT_SMALL   = 20
FONT_TINY    = 16
