from re import S
from manim import *
from manim_slides import Slide
from manim_code_blocks import *
import numpy as np
import random
import math

# CONFIGURACIÓN GLOBAL
FUENTE = "Goudy Old Style"

# PALETA DE COLORES
MARRON_OSCURO = "#3D3834"
NARANJA_TERRACOTA = "#A36536"
PAPEL_CREMA = "#F2E6D8"
PAPEL_TAN = "#B78B68"
FONDO_CAJA = "#FCF3E4"
CAJA_INFERIOR = "#E0C2A8"
TINTA_NEGRA = "#1A1A1A"
SOFT_BG = PAPEL_CREMA
RUST_COLOR = NARANJA_TERRACOTA

RUST_SNIPPETS = {
    "BPETokenizer.rs": """pub struct BPETokenizer {
    vocab: HashMap<String, usize>,
    merges: Vec<(String, String)>,
    unk_token: String,
}""",

    "pair_counts.rs": """let pair_counts: HashMap<(String, String), usize> = tokens
    .par_chunks(chunk_size)
    .enumerate()
    .fold(HashMap::new, |mut local_counts, (_, chunk)| {
        for window in chunk.windows(2) {
            let pair = (window[0].clone(), window[1].clone());
            *local_counts.entry(pair).or_insert(0) += 1;
        }
        local_counts
    })
    .reduce(HashMap::new, |mut a, b| {
        for (k, v) in b { *a.entry(k).or_insert(0) += v; }
        a
    });""",

    "tensor.rs": """pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}""",

    "matmul.rs": """for i in 0..m {
    for j in 0..n {
        let mut sum = 0.0;
        for l in 0..k { sum += a[i * k + l] * b[l * n + j]; }
        result[i * n + j] = sum;
    }
}""",

    "block_matmul.rs": """for i in block_start..block_end {
    for k in k_block_start..k_block_end {
        let a_val = A[i, k];
        for j in j_block_start..j_block_end { 
            C[i, j] += a_val * B[k, j]; 
        }
    }
}""",

    "softmax.rs": """let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
let sum: f32 = exp_vals.iter().sum();
let norm: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();""",

    "embedding.rs": """pub struct Embedding { pub weight: Tensor }

impl Embedding {
    pub fn forward(&self, token_ids: &[Vec<usize>]) -> Tensor {
        let mut out = Vec::with_capacity(batch_size * seq_len * n_embd);
        for batch in token_ids {
            for &id in batch {
                let start = id * n_embd;
                out.extend_from_slice(&self.weight.data[start..start + n_embd]);
            }
        }
        Tensor::new(out, vec![batch_size, seq_len, n_embd])
    }
}""",

    "layernorm.rs": """pub fn forward(&self, x: &Tensor) -> Tensor {
    let mean = x.mean(-1, true);
    let var = x.var(-1, true);
    let norm = x.sub(&mean).div(&var.add_scalar(self.eps).sqrt());
    norm.mul(&self.gamma).add(&self.beta)
}""",

    "attention.rs": """pub fn forward(&self, x: &Tensor) -> Tensor {
    let qkv = self.c_attn.forward(x);
    let (q, k, v) = qkv.split_into_thirds();
    
    let q = self.split_heads(&q);
    let k = self.split_heads(&k);
    let v = self.split_heads(&v);
    
    let scale = 1.0 / (self.head_dim as f32).sqrt();
    let scores = q.matmul(&k.transpose(2, 3)).mul_scalar(scale);
    let mask = self.create_causal_mask(seq_len);
    
    let attn = scores.masked_fill(&mask, f32::NEG_INFINITY).softmax(-1);
    let out = self.merge_heads(&attn.matmul(&v));
    self.c_proj.forward(&out)
}""",

    "gelu.rs": """pub fn gelu(x: &Tensor) -> Tensor {
    let sqrt_2_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
    let res: Vec<f32> = x.data.iter().map(|&val| {
        let inner = sqrt_2_pi * (val + 0.044715_f32 * val.powi(3));
        0.5 * val * (1.0 + inner.tanh())
    }).collect();
    Tensor::new(res, x.shape.clone())
}""",

    "transformer.rs": """pub fn forward(&self, token_ids: &[Vec<usize>]) -> Tensor {
    let seq_len = token_ids[0].len();
    let mut x = self.token_embedding.forward(token_ids);
    x = x.add_broadcast(&self.position_embedding.forward_range(0, seq_len));
    
    for block in &self.blocks { x = block.forward(&x); }
    
    x = self.ln_f.forward(&x);
    self.lm_head.forward(&x)
}""",

    "linear_backward.rs": """pub fn backward(&self, grad_out: &Tensor, cache: &LinearCache) -> LinearGradients {
    LinearGradients { 
        weight: cache.x.transpose(-2, -1).matmul(grad_out), 
        bias: sum_across_positions(grad_out), 
        x: grad_out.matmul(&self.weight.transpose(-2, -1)) 
    }
}""",

    "loss.rs": """pub fn compute_loss(&self, logits: &Tensor, targets: &[usize]) -> f32 {
    for (i, &target) in targets.iter().enumerate() {
        let slice = &logits.data[logit_start..logit_start + vocab_size];
        let max_l = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = slice.iter().map(|&x| (x - max_l).exp()).sum();
        total_loss -= (target_logit - max_l) - exp_sum.ln();
    }
    total_loss / seq_len as f32
}""",

    "layernorm_backward.rs": """pub fn backward(&self, grad_out: &Tensor, cache: &LayerNormCache) -> LayerNormGradients {
    let grad_x_norm = grad_out.mul(&self.gamma);
    for i in 0..seq_len {
        let mean_grad = grad_x_norm_row.iter().sum::<f32>() / n_embd as f32;
        let mean_grad_x = grad_x_norm_row.iter().zip(x_norm_row.iter())
            .map(|(g, x)| g * x).sum::<f32>() / n_embd as f32;
            
        for j in 0..n_embd {
            let diff = grad_x_norm_row[j] - mean_grad;
            grad_x_data[idx] = (diff - x_norm_row[j] * mean_grad_x) / std_val;
        }
    }
    LayerNormGradients { gamma, beta, x: grad_x }
}""",

    "gelu_backward.rs": """pub fn gelu_backward(grad_out: &Tensor, x: &Tensor) -> Tensor {
    let grad_data: Vec<f32> = x.data.par_iter().zip(&grad_out.data)
        .map(|(&x_v, &g_v)| g_v * gelu_derivative(x_v))
        .collect();
    Tensor::new(grad_data, x.shape.clone())
}""",

    "gpt2_backward.rs": """pub fn backward(&self, logits: &Tensor, targs: &[usize], cache: &GPT2Cache) -> GPT2Gradients {
    let grad_logits = compute_cross_entropy_gradient(logits, targs);
    let proj = backprop_output_projection(&grad_logits, cache);
    let mut grad_x = self.ln_final.backward(&proj, &cache.ln_final_cache).x;
    
    let mut block_grads = Vec::new();
    for (block, c) in self.blocks.iter().zip(&cache.block_caches).rev() {
        let grads = block.backward(&grad_x, c);
        grad_x = grads.x.clone();
        block_grads.push(grads);
    }
    
    let (g_tok, g_pos) = backprop_to_embeddings(&grad_x, &cache.input_ids);
    GPT2Gradients { ..Default::default() }
}""",

    "optimizer.rs": """pub fn adamw_update(
    model: &mut TrainableGPT2, grads: &GPT2Gradients, 
    opt: &mut AdamWOptimizer, lr: f32, wd: f32
) {
    opt.step += 1;
    let (bc1, bc2) = (1.0 - beta1.powf(step), 1.0 - beta2.powf(step));
    
    for i in 0..param.len() {
        param[i] *= 1.0 - lr * wd;
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        
        let (m_hat, v_hat) = (m[i] / bc1, v[i] / bc2);
        param[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
    }
}""",

    "train.rs": """let mut loader = TextDataLoader::new(&text, &tok, seq_len, batch_size);
let mut opt = AdamWOptimizer::new(&model);
let mut logger = TrainingLogger::new("log.csv")?;

for step in 0..num_steps {
    let (inputs, targets) = loader.next_batch();
    let (logits, cache) = model.forward(&inputs);
    
    let loss = model.compute_loss(&logits, &targets);
    let mut grads = model.backward(&logits, &targets, &cache);
    
    clip_gradients(&mut grads, 1.0);
    adamw_update(&mut model, &grads, &mut opt, lr, weight_decay);
    
    if step % 50 == 0 { logger.log(step, lr, loss, val_loss, None)?; }
    if step % 500 == 0 { checkpoint.save(&format!("ckpt_{}.bin", step))?; }
}"""
}

C_MADERA_OSCURA = "#3E2723"
C_MADERA_CLARA = "#5D4037"
C_LADRILLO = "#8D6E63"
C_TERRACOTA = "#BF360C"
C_TEJA = "#D84315"
C_HIERRO = "#424242"
C_ACERO = "#78909C"
C_PERGAMINO = "#F4E4BC"
C_TINTA = "#1A1A1A"
C_ORO_VIEJO = "#D4AF37"
C_LATON = "#B5A642"
C_ROJO_SANGRE = "#8B0000"
C_AZUL_NOCHE = "#000080"
C_VERDE_BOSQUE = "#228B22"

def crear_llanuras_manchegas():
    # Usamos colores de tu paleta con opacidades muy bajas para que sean "marca de agua"
    # Ajusta los colores exactos si tienes constantes definidas en tu archivo
    colina_fondo = Ellipse(width=18, height=6, fill_color="#D4B872", fill_opacity=0.06, stroke_width=0)
    colina_fondo.move_to(DOWN * 3.5 + LEFT * 3)
    
    colina_media = Ellipse(width=16, height=4, fill_color="#E6D3A8", fill_opacity=0.08, stroke_width=0)
    colina_media.move_to(DOWN * 3.8 + RIGHT * 3)
    
    colina_frente = Ellipse(width=20, height=3.5, fill_color="#8B5A2B", fill_opacity=0.05, stroke_width=0)
    colina_frente.move_to(DOWN * 4)
    
    # Agrupamos y enviamos al fondo absoluto con z_index
    llanuras = VGroup(colina_fondo, colina_media, colina_frente).set_z_index(-10)
    return llanuras

def crear_molino():
    base_molino = Polygon(
        [-0.45, -0.6, 0], [0.45, -0.6, 0], [0.3, 0.4, 0], [-0.3, 0.4, 0], 
        fill_color=C_LADRILLO, fill_opacity=1, stroke_color=C_MADERA_OSCURA, stroke_width=3
    )
    
    lineas_ladrillo = VGroup(*[
        Line([LEFT_X, -0.4 + i*0.2, 0], [RIGHT_X, -0.4 + i*0.2, 0], color=C_MADERA_OSCURA, stroke_width=1, stroke_opacity=0.5)
        for i, (LEFT_X, RIGHT_X) in enumerate([(-0.4, 0.4), (-0.35, 0.35), (-0.32, 0.32)])
    ])
    
    techo = Polygon(
        [-0.35, 0.4, 0], [0.35, 0.4, 0], [0, 0.9, 0],
        fill_color=C_TEJA, fill_opacity=1, stroke_color=C_MADERA_OSCURA, stroke_width=3
    )
    
    puerta = RoundedRectangle(
        corner_radius=0.1, width=0.3, height=0.4, fill_color=C_MADERA_OSCURA, fill_opacity=1
    ).move_to(base_molino.get_bottom() + UP*0.2)
    
    ventana = Circle(radius=0.08, fill_color=C_TINTA, stroke_color=C_MADERA_OSCURA, stroke_width=2).move_to(base_molino.get_center() + UP*0.1)
    
    def crear_aspa():
        eje = Line(ORIGIN, RIGHT*0.8, stroke_color=C_MADERA_OSCURA, stroke_width=3)
        marco = Rectangle(width=0.6, height=0.25, fill_color=C_PERGAMINO, fill_opacity=0.9, stroke_color=C_MADERA_OSCURA, stroke_width=2)
        marco.next_to(eje.get_start(), RIGHT, buff=0.15)
        entramado = VGroup(
            Line(marco.get_top(), marco.get_bottom(), color=C_MADERA_OSCURA, stroke_width=1),
            Line(marco.get_left(), marco.get_right(), color=C_MADERA_OSCURA, stroke_width=1)
        )
        return VGroup(eje, marco, entramado)

    aspas = VGroup(*[crear_aspa().rotate(i * PI/2, about_point=ORIGIN) for i in range(4)])
    centro = Dot(ORIGIN, color=C_HIERRO, radius=0.08)
    sistema_aspas = VGroup(aspas, centro).move_to(techo.get_bottom() + UP*0.15)
    
    return VGroup(base_molino, lineas_ladrillo, techo, puerta, ventana, sistema_aspas)

def crear_sol_cervantino():
    centro_borde = Circle(radius=0.28, fill_color=C_TERRACOTA, stroke_width=0)
    centro = Circle(radius=0.25, fill_color=C_ORO_VIEJO, stroke_color=C_TERRACOTA, stroke_width=3)
    cara = VGroup(
        Arc(radius=0.1, start_angle=PI, angle=PI, color=C_TERRACOTA, stroke_width=2).shift(DOWN*0.05),
        Dot(radius=0.02, color=C_TERRACOTA).shift(LEFT*0.08 + UP*0.05),
        Dot(radius=0.02, color=C_TERRACOTA).shift(RIGHT*0.08 + UP*0.05)
    )
    
    rayos = VGroup()
    for i in range(16):
        angle = i * (PI / 8)
        length = 0.5 if i % 2 == 0 else 0.35
        rayo = Polygon(
            [0.28, -0.05, 0], [0.28, 0.05, 0], [length, 0, 0],
            fill_color=C_ORO_VIEJO if i % 2 == 0 else C_TERRACOTA, fill_opacity=1, stroke_width=0
        ).rotate(angle, about_point=ORIGIN)
        rayos.add(rayo)
        
    return VGroup(rayos, centro_borde, centro, cara)

def crear_estrella():
    puntas = VGroup()
    for i in range(4):
        angle = i * (PI / 2)
        mitad_clara = Polygon([0,0,0], [0.05,0.05,0], [0,0.25,0], fill_color=C_ORO_VIEJO, fill_opacity=1, stroke_width=0)
        mitad_oscura = Polygon([0,0,0], [-0.05,0.05,0], [0,0.25,0], fill_color=C_LATON, fill_opacity=1, stroke_width=0)
        punta = VGroup(mitad_clara, mitad_oscura).rotate(angle, about_point=ORIGIN)
        puntas.add(punta)
    return puntas

def crear_tintero_y_pluma():
    cuerpo = Polygon(
        [-0.25, -0.2, 0], [0.25, -0.2, 0], [0.2, 0.15, 0], [-0.2, 0.15, 0],
        fill_color=C_TINTA, fill_opacity=1, stroke_color=C_LATON, stroke_width=2
    )
    brillo = Polygon(
        [-0.15, -0.15, 0], [-0.05, -0.15, 0], [-0.05, 0.1, 0], [-0.1, 0.1, 0],
        fill_color=C_ACERO, fill_opacity=0.3, stroke_width=0
    )
    tapa = Rectangle(width=0.3, height=0.08, fill_color=C_LATON, stroke_width=0).next_to(cuerpo, UP, buff=0)
    cuello = Rectangle(width=0.2, height=0.06, fill_color=C_TINTA, stroke_width=0).next_to(tapa, UP, buff=0)
    tintero = VGroup(cuerpo, brillo, tapa, cuello)

    tallo = Line(cuello.get_top(), cuello.get_top() + UP*1.4 + RIGHT*0.8, color=C_PERGAMINO, stroke_width=3)
    punta_pluma = Triangle(fill_color=C_TINTA, stroke_width=0).scale(0.08).move_to(tallo.get_start()).rotate(PI)
    
    pluma_forma = Polygon(
        tallo.get_start() + UP*0.1, tallo.get_start() + UP*0.6 + LEFT*0.3, tallo.get_end() + LEFT*0.1,
        tallo.get_end(), tallo.get_end() + DOWN*0.3 + RIGHT*0.2, tallo.get_start() + RIGHT*0.1 + UP*0.1,
        fill_color=C_PERGAMINO, fill_opacity=1, stroke_color=C_MADERA_CLARA, stroke_width=1
    )
    cortes = VGroup(
        Line(tallo.get_start() + UP*0.4 + LEFT*0.15, tallo.get_start() + UP*0.35, color=C_MADERA_CLARA, stroke_width=2),
        Line(tallo.get_end() + DOWN*0.4, tallo.get_end() + DOWN*0.3 + RIGHT*0.1, color=C_MADERA_CLARA, stroke_width=2)
    )
    
    gotas = VGroup(
        Dot(radius=0.04, color=C_TINTA).move_to(cuello.get_right() + RIGHT*0.2 + UP*0.2),
        Dot(radius=0.02, color=C_TINTA).move_to(cuello.get_right() + RIGHT*0.35 + UP*0.05),
        Dot(radius=0.03, color=C_TINTA).move_to(cuello.get_left() + LEFT*0.1 + UP*0.1)
    )
    
    return VGroup(tintero, pluma_forma, cortes, tallo, punta_pluma, gotas)

def crear_pila_libros():
    def crear_libro(ancho, alto, color_cubierta, color_paginas, rotacion, desplazamiento):
        cubierta = RoundedRectangle(corner_radius=0.05, width=ancho, height=alto, fill_color=color_cubierta, fill_opacity=1, stroke_color=C_TINTA, stroke_width=2)
        paginas = Rectangle(width=ancho-0.1, height=alto-0.08, fill_color=color_paginas, stroke_width=0).move_to(cubierta).align_to(cubierta, RIGHT).shift(LEFT*0.02)
        lineas_pag = VGroup(*[Line(paginas.get_left() + UP*(alto/2 - 0.08 - i*0.04), paginas.get_right() + UP*(alto/2 - 0.08 - i*0.04), color=C_MADERA_CLARA, stroke_width=1) for i in range(int(alto/0.05))])
        lomo_detalle = Line(cubierta.get_left() + UP*(alto/3), cubierta.get_left() + DOWN*(alto/3), color=C_ORO_VIEJO, stroke_width=2).shift(RIGHT*0.15)
        libro = VGroup(cubierta, paginas, lineas_pag, lomo_detalle).rotate(rotacion).shift(desplazamiento)
        return libro

    libro1 = crear_libro(1.5, 0.35, C_ROJO_SANGRE, C_PERGAMINO, 0, ORIGIN)
    libro2 = crear_libro(1.3, 0.3, C_AZUL_NOCHE, C_PERGAMINO, 2 * DEGREES, UP*0.32 + LEFT*0.05)
    marcapaginas = Rectangle(width=0.08, height=0.5, fill_color=C_ORO_VIEJO, stroke_width=1, stroke_color=C_TINTA).move_to(libro2.get_right() + LEFT*0.3 + DOWN*0.2)
    libro3 = crear_libro(1.2, 0.25, C_VERDE_BOSQUE, C_PERGAMINO, -8 * DEGREES, UP*0.58 + RIGHT*0.05)

    return VGroup(libro1, marcapaginas, libro2, libro3)

def crear_escudo_y_lanza():
    astil = Line(DOWN*1.2 + LEFT*0.6, UP*1.2 + RIGHT*0.6, color=C_MADERA_OSCURA, stroke_width=6)
    punta_base = Polygon(astil.get_end() + LEFT*0.1 + DOWN*0.05, astil.get_end() + RIGHT*0.1 + DOWN*0.05, astil.get_end() + UP*0.3 + RIGHT*0.15, fill_color=C_ACERO, stroke_color=C_HIERRO, stroke_width=2)
    
    escudo_borde = Circle(radius=0.55, fill_color=C_HIERRO, stroke_width=0)
    escudo_fondo = Circle(radius=0.45, fill_color=C_MADERA_CLARA, stroke_width=0)
    cruz_v = Rectangle(width=0.15, height=0.9, fill_color=C_TERRACOTA, stroke_width=0)
    cruz_h = Rectangle(width=0.9, height=0.15, fill_color=C_TERRACOTA, stroke_width=0)
    
    remaches = VGroup(*[Dot(radius=0.03, color=C_ACERO).move_to(escudo_borde.point_at_angle(i * PI/6)).shift(IN*0.05) for i in range(12)])
    
    escudo = VGroup(escudo_borde, escudo_fondo, cruz_v, cruz_h, remaches).move_to(astil.get_center() + DOWN*0.2 + RIGHT*0.2)
    return VGroup(astil, punta_base, escudo)

def crear_herradura():
    cuerpo = AnnularSector(
        inner_radius=0.25, outer_radius=0.45, angle=PI*1.4, start_angle=-PI*0.2, 
        fill_color=C_HIERRO, fill_opacity=1, stroke_color=C_TINTA, stroke_width=2
    ).rotate(-PI/2 - PI*0.2)
    
    borde_interior = AnnularSector(
        inner_radius=0.32, outer_radius=0.38, angle=PI*1.3, start_angle=-PI*0.15,
        fill_color=C_TINTA, fill_opacity=0.3, stroke_width=0
    ).rotate(-PI/2 - PI*0.15)
    
    agujeros = VGroup()
    for i in range(7):
        angulo = i * (PI/5) + PI*0.1
        punto = Rectangle(width=0.02, height=0.06, fill_color=C_TINTA, stroke_width=0).move_to(
            [0.35 * np.cos(angulo), 0.35 * np.sin(angulo), 0]
        ).rotate(angulo + PI/2)
        agujeros.add(punto)
    
    herradura = VGroup(cuerpo, borde_interior, agujeros).rotate(-PI*0.2)
    return herradura

def crear_rueda_carreta():
    aro_hierro = Circle(radius=0.6, stroke_color=C_HIERRO, stroke_width=8)
    aro_madera = Circle(radius=0.54, stroke_color=C_MADERA_CLARA, stroke_width=12)
    centro_madera = Circle(radius=0.15, fill_color=C_MADERA_OSCURA, stroke_color=C_HIERRO, stroke_width=3)
    eje = Dot(radius=0.05, color=C_HIERRO)
    
    radios = VGroup(*[
        Polygon([-0.03, 0.15, 0], [0.03, 0.15, 0], [0.02, 0.5, 0], [-0.02, 0.5, 0], fill_color=C_MADERA_CLARA, stroke_color=C_MADERA_OSCURA, stroke_width=1).rotate(i * PI/6, about_point=ORIGIN)
        for i in range(12)
    ])
    
    remaches = VGroup(*[Dot(radius=0.02, color=C_ACERO).move_to(aro_hierro.point_at_angle(i * PI/6)) for i in range(12)])
    
    return VGroup(radios, aro_madera, aro_hierro, remaches, centro_madera, eje)

def crear_pergamino():
    cuerpo = Polygon(
        [-0.4, 0.6, 0], [0.4, 0.6, 0], [0.35, -0.6, 0], [-0.45, -0.6, 0],
        fill_color=C_PERGAMINO, fill_opacity=1, stroke_color=C_MADERA_OSCURA, stroke_width=2
    )
    
    rollo_sup = RoundedRectangle(corner_radius=0.1, width=1.0, height=0.25, fill_color=C_PERGAMINO, fill_opacity=1, stroke_color=C_MADERA_OSCURA, stroke_width=2).move_to(UP*0.6)
    rollo_sup_espiral = Arc(radius=0.08, start_angle=0, angle=PI*1.5, color=C_MADERA_OSCURA, stroke_width=2).move_to(rollo_sup.get_right() + LEFT*0.1)
    
    rollo_inf = RoundedRectangle(corner_radius=0.1, width=0.9, height=0.25, fill_color=C_PERGAMINO, fill_opacity=1, stroke_color=C_MADERA_OSCURA, stroke_width=2).move_to(DOWN*0.6)
    rollo_inf_espiral = Arc(radius=0.08, start_angle=PI, angle=PI*1.5, color=C_MADERA_OSCURA, stroke_width=2).move_to(rollo_inf.get_left() + RIGHT*0.1)
    
    sello_cera = Circle(radius=0.15, fill_color=C_ROJO_SANGRE, stroke_color=C_TINTA, stroke_width=1).move_to(cuerpo.get_bottom() + UP*0.2 + RIGHT*0.2)
    sello_detalle = Circle(radius=0.1, stroke_color=C_TINTA, stroke_width=1).move_to(sello_cera)
    
    lineas = VGroup()
    anchos = [0.6, 0.7, 0.5, 0.65, 0.4]
    for i, ancho in enumerate(anchos):
        linea = Line(LEFT*(ancho/2), RIGHT*(ancho/2), color=C_TINTA, stroke_width=2).shift(UP*(0.3 - i*0.15))
        lineas.add(linea)
        
    return VGroup(cuerpo, rollo_sup, rollo_sup_espiral, rollo_inf, rollo_inf_espiral, lineas, sello_cera, sello_detalle)

def crear_yelmo_mambrino():
    cuenco = Arc(
        radius=0.5, start_angle=0, angle=PI, 
        fill_color=C_LATON, fill_opacity=1, stroke_color=C_MADERA_OSCURA, stroke_width=2
    )
    
    brillo = Arc(
        radius=0.4, start_angle=PI/4, angle=PI/3,
        color=C_PERGAMINO, stroke_width=4, stroke_opacity=0.6
    )
    
    borde = RoundedRectangle(
        corner_radius=0.08, width=1.5, height=0.15, 
        fill_color=C_LATON, fill_opacity=1, stroke_color=C_MADERA_OSCURA, stroke_width=2
    ).next_to(cuenco, DOWN, buff=-0.05)
    
    muesca = Arc(
        radius=0.18, start_angle=0, angle=PI, 
        fill_color=WHITE, fill_opacity=1, stroke_width=2, stroke_color=C_MADERA_OSCURA
    ).move_to(borde.get_bottom() + UP*0.08)
    
    return VGroup(cuenco, borde, brillo, muesca)

class Presentacion(Slide):
    def construct(self):
        self.camera.background_color = WHITE

        self.slide_introduction()
        self.slide_credits()
        self.slide_que_es_transformer()
        self.slide_molinete_ai()
        self.slide_por_que_rust()

        self.slide_que_es_un_tensor()
        self.slide_strides()
        self.slide_matmul()
        self.slide_softmax()

        self.slide_forward_pass()
        self.diapo_problema_strawberry()
        self.diapo_tokenizacion()
        self.diapo_byte_pair_encoding()
        self.diapo_tamano_vocabulario()
        self.slide_embeddings()
        self.slide_position_embeddings()
        self.slide_layer_normalization()

        self.slide_mha_acto1_intuicion()
        self.slide_mha_acto2_formula()
        self.slide_mha_acto3_calculo()
        self.slide_mha_acto4_multihead()

        self.mostrar_acto_arquitectura_neurona()
        self.mostrar_acto_zoom_neurona()
        self.slide_residual()
        self.mostrar_acto_activacion()
        self.slide_capa_transformer()


        self.slide_entrenamiento()
        self.slide_descenso_gradiente()
        self.slide_backpropagation()
        self.slide_adam()
        self.slide_dropout()
        
        self.slide_training_metrics()
        self.slide_temperature()

        self.slide_model_in_action()
        self.slide_final()
                
    # --- FUNCIONES AUXILIARES ---

    def mostrar_snippet(self, titulo_archivo):
        self.diapo_codigo(
            codigo_fuente=RUST_SNIPPETS[titulo_archivo],
            titulo_archivo=titulo_archivo
        )
        self.limpiar_pantalla()

    def crear_titulo(self, texto, palabra_clave=None, color_clave=NARANJA_TERRACOTA, font_size=35):
        t2c = {palabra_clave: color_clave} if palabra_clave else {}
        titulo = Text(texto, font=FUENTE, font_size=font_size, color=TINTA_NEGRA, t2c=t2c).to_edge(UP)
        linea = Underline(titulo, color=color_clave, stroke_width=4)
        return titulo, linea

    def crear_bloque(self, texto="", color_fondo=FONDO_CAJA, color_texto=TINTA_NEGRA, ancho=0.8, alto=0.8):
        rect = RoundedRectangle(
            corner_radius=0.15, width=ancho, height=alto, 
            fill_color=color_fondo, fill_opacity=1, 
            stroke_color=MARRON_OSCURO, stroke_width=2
        )
        lbl = Text(str(texto), font=FUENTE, font_size=24, color=color_texto).move_to(rect.get_center())
        return VGroup(rect, lbl)
    
    def crear_matriz_bloques(self, filas, columnas, color_fondo=FONDO_CAJA, color_texto=TINTA_NEGRA, valores=None, ancho=0.8, alto=0.8):
        if valores is None:
            valores = [""] * (filas * columnas)
            
        matriz = VGroup()
        idx = 0
        
        for i in range(filas):
            fila_bloques = VGroup()
            for j in range(columnas):
                texto = valores[idx] if idx < len(valores) else ""

                bloque = self.crear_bloque(
                    texto=texto, 
                    color_fondo=color_fondo, 
                    color_texto=color_texto, 
                    ancho=ancho, 
                    alto=alto
                )
                fila_bloques.add(bloque)
                idx += 1
                
            fila_bloques.arrange(RIGHT, buff=0.05)
            matriz.add(fila_bloques)
            
        matriz.arrange(DOWN, buff=0.05)
        
        return matriz

    def limpiar_pantalla(self):
        if self.mobjects:
            self.play(*[FadeOut(mob) for mob in self.mobjects])

    # --- DIAPOSITIVAS ---

    def slide_introduction(self):

        titulo, linea = self.crear_titulo("Construyendo un Transformer con Rust", palabra_clave="Rust")
        subtitulo = Text("Jerónimo Hoyos Botero", font=FUENTE, font_size=25, color=MARRON_OSCURO)
        VGroup(titulo, linea, subtitulo).arrange(DOWN, buff=0.2).to_edge(UP, buff=0.5)

        escritorio_decor = crear_tintero_y_pluma().scale(0.8).next_to(titulo, RIGHT, buff=0.4).shift(DOWN*0.2)
        sol_decor = crear_sol_cervantino().scale(0.7).to_corner(UR, buff=0.5)
        
        molino_grande = crear_molino().scale(0.8).to_corner(DR, buff=0.5).shift(UP*0.5 + LEFT*0.2)
        molino_pequeño = crear_molino().scale(0.5).next_to(molino_grande, LEFT, buff=0.1).shift(DOWN*0.3 + RIGHT*0.2)
        molinos_paisaje = VGroup(molino_pequeño, molino_grande)
        
        molino_grande[3].add_updater(lambda m, dt: m.rotate(-dt * 0.8))
        molino_pequeño[3].add_updater(lambda m, dt: m.rotate(-dt * 0.5))
        
        libros_decor = crear_pila_libros().scale(0.9).to_corner(DL, buff=0.6).shift(UP*0.5)
        armas_decor = crear_escudo_y_lanza().scale(0.8).next_to(libros_decor, RIGHT, buff=0.2).shift(DOWN*0.1)

        estrellas = VGroup(
            crear_estrella().move_to(escritorio_decor.get_top() + UP*0.3 + RIGHT*0.2),
            crear_estrella().scale(0.6).move_to(escritorio_decor.get_bottom() + RIGHT*0.6),
            crear_estrella().scale(0.8).move_to(armas_decor.get_top() + UP*0.2 + LEFT*0.2),
            crear_estrella().scale(0.5).move_to(molinos_paisaje.get_top() + UP*0.4 + LEFT*0.5)
        )

        cajas = VGroup(*[
            RoundedRectangle(
                corner_radius=0.1, width=2.6, height=1.8, fill_color=FONDO_CAJA, fill_opacity=1, 
                stroke_color=MARRON_OSCURO, stroke_width=2
            ).shift(UP * i * 0.06 + RIGHT * i * 0.06) for i in range(5)
        ]).move_to(LEFT * 5.5 + DOWN * 0.5) 
        
        txt_molinete = Text("Molinete AI", font=FUENTE, font_size=20, weight=BOLD, color=TINTA_NEGRA).move_to(cajas)
        flecha = Arrow(LEFT, RIGHT, color=TINTA_NEGRA, stroke_width=4, max_tip_length_to_length_ratio=0.15).scale(0.6).next_to(cajas, RIGHT, buff=0.3)

        pos_start = flecha.get_right() + RIGHT * 4.3 + UP * 1.6 

        def create_probs(target_word):
            dummies = ["hidalgo", "espada", "vino", "capa", "oro", "plaza", "mujer", "caballo"]
            random.shuffle(dummies)
            probs = [random.uniform(0.7, 0.9), random.uniform(0.05, 0.1), random.uniform(0.01, 0.03)]

            datos = [(target_word, probs[0], NARANJA_TERRACOTA), (dummies[0], probs[1], PAPEL_TAN), (dummies[1], probs[2], CAJA_INFERIOR)]
            txts = VGroup(*[Text(d[0], font=FUENTE, font_size=16, color=TINTA_NEGRA) for d in datos]).arrange(DOWN, aligned_edge=RIGHT, buff=0.15)
            bars = VGroup(*[RoundedRectangle(corner_radius=0.05, height=0.15, width=max(0.1, d[1]*2), fill_color=d[2], fill_opacity=1, stroke_width=0) for d in datos]).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            
            for i in range(len(datos)): bars[i].match_y(txts[i])
            
            col = VGroup(txts, bars).arrange(RIGHT, buff=0.15)
            bg = RoundedRectangle(
                corner_radius=0.1, width=col.width+0.5, height=txts[0].height+0.3, 
                fill_color=CAJA_INFERIOR, fill_opacity=0.3, stroke_color=MARRON_OSCURO, stroke_width=1
            ).move_to(txts[0]).match_x(col).shift(RIGHT*0.1)
            
            return VGroup(bg, col).move_to(flecha.get_right() + RIGHT * 0.4, aligned_edge=LEFT)
        
        self.play(
            Write(titulo), Create(linea), FadeIn(subtitulo, shift=DOWN),
            DrawBorderThenFill(escritorio_decor, run_time=1.5),
            SpinInFromNothing(sol_decor),
            GrowFromCenter(molinos_paisaje),
            FadeIn(libros_decor, shift=UP*0.5),
            DrawBorderThenFill(armas_decor, run_time=1.2),
            Create(estrellas, lag_ratio=0.2)
        )
        
        for estrella in estrellas:
            estrella.add_updater(lambda m, dt: m.rotate(dt * 0.5))

        self.next_slide()
        
        self.play(FadeIn(cajas, shift=RIGHT*0.5), Write(txt_molinete), GrowArrow(flecha))

        poema = [
            ["retorciendo", "el", "mostacho", "soldadesco,"],
            ["por", "ver", "que", "ya", "su", "bolsa", "le", "repica,"],
            ["a", "un", "corrillo", "llegó", "de", "gente", "rica"],
            ["y", "en", "el", "nombre", "de", "Dios", "pidió", "refresco."],
            ["Den", "voacedes,", "por", "Dios,", "a", "mi", "pobreza,"],
            ["les", "dice;", "donde", "no,", "por", "ocho", "santos"],
            ["que", "haré", "lo", "que", "hacer", "suelo", "sin", "tardanza."]
        ]
        
        curr_y = pos_start[1]
        curr_probs = create_probs("...").set_opacity(0)

        for linea_texto in poema:
            curr_x = pos_start[0]
            for word in linea_texto:
                new_probs = create_probs(word)
                word_mob = Text(word, font=FUENTE, font_size=20, color=TINTA_NEGRA).move_to([curr_x, curr_y, 0], aligned_edge=LEFT)
                
                self.play(ReplacementTransform(curr_probs, new_probs), FadeIn(word_mob, shift=LEFT*0.1), run_time=0.45)
                curr_probs, curr_x = new_probs, curr_x + word_mob.width + 0.15
            curr_y -= 0.45

        molino_grande[3].clear_updaters()
        molino_pequeño[3].clear_updaters()
        for estrella in estrellas:
            estrella.clear_updaters()

        self.next_slide()
        self.limpiar_pantalla()

    def slide_credits(self):
        titulo_creditos, linea_creditos = self.crear_titulo(
            "Esta presentación se basa en:", 
            palabra_clave="basa en:", 
            color_clave=MARRON_OSCURO 
        )
        self.play(Write(titulo_creditos), Create(linea_creditos))

        imagen_creditos = ImageMobject("assets/creditos_guia_original.png")
        imagen_creditos.scale(1.5)
        imagen_creditos.next_to(linea_creditos, DOWN, buff=0.8)
        molino = crear_molino().to_corner(DL).shift(UP * 0.5 + RIGHT * 0.5)
        sol = crear_sol_cervantino().to_corner(UR).shift(DOWN * 0.5 + LEFT * 0.5)
        escudo = crear_escudo_y_lanza().to_corner(UL).shift(DOWN * 0.2 + RIGHT * 0.5)
        libros = crear_pila_libros().to_corner(DR).shift(UP * 0.5 + LEFT * 0.5)
        tintero = crear_tintero_y_pluma().next_to(libros, LEFT, buff=0.8)
        

        estrellas = VGroup(
            crear_estrella().move_to(UP * 1.5 + LEFT * 4),
            crear_estrella().move_to(DOWN * 1.5 + LEFT * 5),
            crear_estrella().move_to(UP * 0.5 + RIGHT * 5)
        )

        self.play(
            FadeIn(imagen_creditos, shift=UP),
            FadeIn(molino, shift=RIGHT),
            SpinInFromNothing(sol),
            FadeIn(escudo, shift=RIGHT),
            FadeIn(libros, shift=DOWN),
            FadeIn(tintero, shift=DOWN),
            FadeIn(estrellas),
            run_time=2
        )

        def crear_corazon_pirotecnico(centro, color, escala=2.5, num_particulas=200):
            """
            Crea un corazón de fuego artificial más lindo y definido.
            - `escala`: Controla el tamaño general de CADA corazón.
            - `num_particulas`: Más partículas para formas densas y lindas.
            """
            radius = 0.03 
            particulas = VGroup()
            animaciones = []

            for i in range(num_particulas):
                t = i * 2 * np.pi / num_particulas
                

                x = 16 * np.sin(t)**3
                y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
                
                destino_vec = np.array([x, y, 0]) * (escala * 0.08)
                pos_inicio = centro + np.random.uniform(-0.02, 0.02, 3) 

                particula = Dot(pos_inicio, radius=radius, color=color, fill_opacity=1.0)
                particula.set_stroke(color=color, width=0.5, opacity=0.8) 


                pos_final = centro + destino_vec + np.random.uniform(-0.15, 0.15, 3)

                particulas.add(particula)

                animaciones.append(
                    particula.animate(rate_func=rate_functions.ease_out_expo).move_to(pos_final).set_opacity(0).scale(0.01)
                )

            return particulas, AnimationGroup(*animaciones)

        num_explosiones = 8
        fuegos_grupos = []
        animaciones_conjunto = []

        safe_x = 7.0 
        safe_y_top = 3.5
        safe_y_bottom = -3.5

        for _ in range(num_explosiones):
            x = np.random.uniform(-safe_x, safe_x)
            y = np.random.uniform(safe_y_bottom, safe_y_top)

            if y > titulo_creditos.get_bottom()[1] - 0.5:
                y -= 1.0 

            pos = np.array([x, y, 0])

            color = np.random.choice([NARANJA_TERRACOTA, MARRON_OSCURO]) 
            escala = np.random.uniform(2.0, 3.5) 
            random_lag = np.random.uniform(0.1, 0.6) 

            particulas_gp, anim_gp = crear_corazon_pirotecnico(
                centro=pos,
                color=color,
                escala=escala
            )
            
            fuegos_grupos.append(particulas_gp)

            animaciones_conjunto.append(
                AnimationGroup(anim_gp, lag_ratio=0.0)
            )
        self.next_slide()
        self.add(*fuegos_grupos) 

        self.play(
            LaggedStart(*animaciones_conjunto, lag_ratio=0.1), 
            run_time=6.0 
        )

        self.remove(*fuegos_grupos)
        self.limpiar_pantalla()

    def slide_que_es_transformer(self):
        titulo, linea = self.crear_titulo("¿Qué es un Transformer?", palabra_clave="Transformer", color_clave=NARANJA_TERRACOTA)

        molino = crear_molino().scale(0.8).to_corner(DL).shift(UP*0.5 + RIGHT*0.5)
        sol = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.5 + LEFT*0.5)
        tintero = crear_tintero_y_pluma().scale(0.8).to_corner(DR).shift(UP*0.5 + LEFT*0.5)
        escudo = crear_escudo_y_lanza().scale(0.8).to_corner(UL).shift(DOWN*0.5 + RIGHT*0.5)
        adornos = VGroup(molino, sol, tintero, escudo)

        COLOR_PRED = NARANJA_TERRACOTA 
        
        explicacion_1 = Text(
            "TRANSFORMA la secuencia, observa todo el contexto\n"
            "para predecir qué palabra tiene más sentido a continuación.", 
            font=FUENTE, font_size=26, color=TINTA_NEGRA, line_spacing=1.2,
            t2c={"TRANSFORMA": NARANJA_TERRACOTA, "todo el contexto": MARRON_OSCURO, "tiene más sentido": COLOR_PRED}
        ).move_to(UP * 2.2)

        caja_maq = RoundedRectangle(corner_radius=0.2, width=2.5, height=1.5, fill_color=MARRON_OSCURO, fill_opacity=0.9, stroke_color=NARANJA_TERRACOTA, stroke_width=3)
        txt_maq = Text("Modelo IA", font=FUENTE, font_size=24, color=FONDO_CAJA, weight=BOLD).move_to(caja_maq)
        maquina = VGroup(caja_maq, txt_maq).to_edge(LEFT, buff=1.0).shift(DOWN * 0.5)


        def crear_burbuja_chat(texto, es_usuario=True):
            color_fondo = CAJA_INFERIOR if es_usuario else FONDO_CAJA
            color_borde = MARRON_OSCURO if es_usuario else NARANJA_TERRACOTA
            txt = Text(texto, font=FUENTE, font_size=22, color=TINTA_NEGRA)
            burbuja = RoundedRectangle(
                width=max(txt.width + 0.6, 1.8), height=txt.height + 0.5, corner_radius=0.2,
                fill_color=color_fondo, fill_opacity=1, stroke_color=color_borde, stroke_width=2
            )
            txt.move_to(burbuja.get_center())
            return VGroup(burbuja, txt)

        def animar_pensamiento(texto_ganador, texto_perdedor, pct_ganador, pct_perdedor, obj_referencia):
            opcion_ganadora = Text(f"{texto_ganador} ({pct_ganador})", font=FUENTE, font_size=20, weight=BOLD, color=COLOR_PRED)
            opcion_perdedora = Text(f"{texto_perdedor} ({pct_perdedor})", font=FUENTE, font_size=16, color=GRAY)
            textos_menu = VGroup(opcion_ganadora, opcion_perdedora).arrange(DOWN, buff=0.15)
            caja_menu = SurroundingRectangle(textos_menu, color=COLOR_PRED, stroke_width=2, fill_color=WHITE, fill_opacity=0.95, corner_radius=0.1, buff=0.25)
            menu_final = VGroup(caja_menu, textos_menu).next_to(obj_referencia, RIGHT, buff=0.5)

            self.play(Wiggle(obj_referencia))
            self.play(FadeIn(menu_final, shift=RIGHT * 0.3))
            self.play(Indicate(opcion_ganadora, scale_factor=1.2, color=NARANJA_TERRACOTA))
            self.play(FadeOut(menu_final))

        chat_x_right = RIGHT * 5.2 
        chat_x_left = RIGHT * 1.5   
        
        self.play(Write(titulo), Create(linea), FadeIn(adornos, lag_ratio=0.2), run_time=1.5)
        self.play(FadeIn(explicacion_1, shift=UP * 0.2), FadeIn(maquina, shift=RIGHT * 0.5))
        self.next_slide()

        # Interacción 1
        msg1_usuario = crear_burbuja_chat("¿Quién es?", es_usuario=True).move_to(chat_x_right + UP * 0.8).align_to(chat_x_right, RIGHT)
        self.play(FadeIn(msg1_usuario, shift=UP * 0.2))
        
        animar_pensamiento("Soy yo", "El cartero", "98.5%", "1.5%", caja_maq) 
        
        msg2_ia = crear_burbuja_chat("Soy yo", es_usuario=False).next_to(msg1_usuario, DOWN, buff=0.3).align_to(chat_x_left, LEFT)
        self.play(FadeIn(msg2_ia, shift=UP * 0.2))
        self.next_slide()

        # Interacción 2
        msg3_usuario = crear_burbuja_chat("¿Qué vienes a buscar?", es_usuario=True).next_to(msg2_ia, DOWN, buff=0.3).align_to(chat_x_right, RIGHT)
        self.play(FadeIn(msg3_usuario, shift=UP * 0.2))
        
        animar_pensamiento("A ti", "Nada", "99.9%", "0.05%", caja_maq) 
        
        msg_final_ia = crear_burbuja_chat("A ti", es_usuario=False).next_to(msg3_usuario, DOWN, buff=0.3).align_to(chat_x_left, LEFT)
        self.play(FadeIn(msg_final_ia, shift=UP * 0.2))
        self.next_slide()

        shift_up = UP * (msg1_usuario.get_center()[1] - msg3_usuario.get_center()[1])
        
        self.play(
            FadeOut(msg1_usuario, shift=shift_up),
            FadeOut(msg2_ia, shift=shift_up),
            msg3_usuario.animate.shift(shift_up),
            msg_final_ia.animate.shift(shift_up),
            run_time=1.2
        )

        # Interacción 3
        msg5_usuario = crear_burbuja_chat("Ya es tarde", es_usuario=True).next_to(msg_final_ia, DOWN, buff=0.3).align_to(chat_x_right, RIGHT)
        self.play(FadeIn(msg5_usuario, shift=UP * 0.2))
        
        animar_pensamiento("¿Por qué?", "Vete", "95.0%", "5.0%", caja_maq) 
        
        msg6_ia = crear_burbuja_chat("¿Por qué?", es_usuario=False).next_to(msg5_usuario, DOWN, buff=0.3).align_to(chat_x_left, LEFT)
        self.play(FadeIn(msg6_ia, shift=UP * 0.2))
        self.next_slide()

        shift_up_2 = UP * (msg3_usuario.get_center()[1] - msg5_usuario.get_center()[1])
        
        self.play(
            FadeOut(msg3_usuario, shift=shift_up_2),
            FadeOut(msg_final_ia, shift=shift_up_2),
            msg5_usuario.animate.shift(shift_up_2),
            msg6_ia.animate.shift(shift_up_2),
            run_time=1.2
        )

        texto_despecho = "Porque ahora soy yo la que\nquiere estar sin ti"
        msg7_usuario = crear_burbuja_chat(texto_despecho, es_usuario=True).next_to(msg6_ia, DOWN, buff=0.3).align_to(chat_x_right, RIGHT)
        self.play(FadeIn(msg7_usuario, shift=UP * 0.2))
        self.next_slide()

        titulo_utilidades = Text("¿Para qué sirven?", font=FUENTE, font_size=32, color=MARRON_OSCURO, weight=BOLD).move_to(UP * 1.5)

        def crear_item_utilidad(titulo_item, descripcion):
            t_titulo = Text(titulo_item, font=FUENTE, font_size=26, color=NARANJA_TERRACOTA, weight=BOLD)
            t_desc = Text(descripcion, font=FUENTE, font_size=22, color=TINTA_NEGRA)
            grupo_texto = VGroup(t_titulo, t_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            marcador = Rectangle(width=0.08, height=grupo_texto.height, fill_color=MARRON_OSCURO, fill_opacity=1, stroke_width=0)
            return VGroup(marcador, grupo_texto).arrange(RIGHT, buff=0.25)

        lista_utilidades = VGroup(
            crear_item_utilidad("1. Chatbots Asistentes", "Conversación natural fluida"),
            crear_item_utilidad("2. Agentes Autónomos", "Resolución de tareas complejas"),
            crear_item_utilidad("3. Predicción Avanzada", "Código, traducción y análisis")
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.6).move_to(DOWN * 0.5)

        chat_visible_final = VGroup(msg5_usuario, msg6_ia, msg7_usuario)
        
        self.play(
            FadeOut(chat_visible_final), 
            FadeOut(maquina), 
            FadeOut(explicacion_1), 
            run_time=1
        )
        self.play(FadeIn(titulo_utilidades, shift=UP * 0.2))
        self.play(AnimationGroup(*[FadeIn(item, shift=RIGHT * 0.3) for item in lista_utilidades], lag_ratio=0.3))
        self.next_slide()
        
        self.limpiar_pantalla()

    def slide_molinete_ai(self):
        titulo, linea = self.crear_titulo(
            "Molinete AI", 
            palabra_clave="Molinete", 
            color_clave=NARANJA_TERRACOTA
        )

        molino_adorno = crear_molino().scale(0.8).to_corner(DL, buff=0.8)
        sol_adorno = crear_sol_cervantino().scale(0.8).to_corner(UR, buff=0.8)
        tintero_adorno = crear_tintero_y_pluma().scale(0.8).to_corner(DR, buff=0.8)
        escudo_adorno = crear_escudo_y_lanza().scale(0.8).to_corner(UL, buff=0.8)
        
        adornos = VGroup(molino_adorno, sol_adorno, tintero_adorno, escudo_adorno)

        subtitulo = Text(
            "Origen del nombre “Molinete”", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA, weight=BOLD
        )

        descripcion = Text(
            "Modelo de lenguaje entrenado sobre\nel corpus de El Quijote.", 
            font=FUENTE, font_size=24, color=MARRON_OSCURO
        )

        punto1 = Text(
            "• Inspirado en el imaginario cervantino.",
            font=FUENTE, font_size=22, color=TINTA_NEGRA, 
            t2c={"imaginario cervantino": NARANJA_TERRACOTA}
        )

        punto2 = Text(
            "• Nombre que evoca los molinos de viento\n  enfrentados por Don Quijote.",
            font=FUENTE, font_size=22, color=TINTA_NEGRA,
            t2c={"molinos de viento": NARANJA_TERRACOTA}
        )

        punto3 = Text(
            "• Identidad literaria para un modelo\n  de lenguaje contemporáneo.",
            font=FUENTE, font_size=22, color=TINTA_NEGRA,
            t2c={"modelo de lenguaje": MARRON_OSCURO}
        )

        grupo_puntos = VGroup(punto1, punto2, punto3).arrange(DOWN, aligned_edge=LEFT, buff=0.4) 
        textos_molinete = VGroup(descripcion, subtitulo, grupo_puntos).arrange(DOWN, aligned_edge=LEFT, buff=0.5) 

        imagen_molino = ImageMobject("assets/quijote_vs_molinos.png")
        imagen_molino.height = 5.5 
        contenido_principal = Group(textos_molinete, imagen_molino).arrange(RIGHT, buff=0.8)

        contenido_principal.next_to(linea, DOWN, buff=0.5)

        if contenido_principal.width > 12.5:
            contenido_principal.width = 12.5
            contenido_principal.next_to(linea, DOWN, buff=0.5) 

        self.play(Write(titulo), Create(linea), FadeIn(adornos, lag_ratio=0.2))
        self.play(FadeIn(descripcion, shift=UP * 0.2))
        self.next_slide()

        self.play(
            FadeIn(subtitulo, shift=RIGHT * 0.4),
            FadeIn(imagen_molino, shift=LEFT * 0.4)
        )
        self.next_slide()
        
        self.play(
            LaggedStart(
                FadeIn(punto1, shift=UP * 0.2),
                FadeIn(punto2, shift=UP * 0.2),
                FadeIn(punto3, shift=UP * 0.2),
                lag_ratio=0.8
            ),
            run_time=2.5
        )
        self.next_slide()

        self.limpiar_pantalla()

    def diapo_problema_strawberry(self):
        titulo, linea = self.crear_titulo(
            "¿Por qué los LLM no saben 'leer'?", 
            palabra_clave="'leer'?", 
            color_clave=NARANJA_TERRACOTA
        )
        self.limpiar_pantalla()
        
        llanuras_fondo = crear_llanuras_manchegas()
        
        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)

        self.play(
            Write(titulo), 
            Create(linea),
            FadeIn(llanuras_fondo, shift=UP),
            FadeIn(sol_fondo)
        )
        self.next_slide()

        def crear_burbuja(texto, color_fondo, color_texto, es_usuario=True, t2c_dict=None):
            txt = Text(texto, font=FUENTE, font_size=24, color=color_texto, t2c=t2c_dict)
            
            fondo = RoundedRectangle(
                width=txt.width + 0.8, 
                height=txt.height + 0.5, 
                corner_radius=0.2, 
                fill_color=color_fondo, fill_opacity=1, 
                stroke_width=0 if es_usuario else 1.5,
                stroke_color=MARRON_OSCURO
            )
            sombra = fondo.copy().set_fill(MARRON_OSCURO, 0.1).set_stroke(width=0).shift(RIGHT*0.05 + DOWN*0.05)
            
            txt.move_to(fondo.get_center())
            burbuja_base = VGroup(sombra, fondo, txt)
            
            remitente = Text(
                "Tú" if es_usuario else "Molinete AI", 
                font=FUENTE, font_size=16, 
                color=MARRON_OSCURO, weight=BOLD
            )
            
            if es_usuario:
                remitente.next_to(burbuja_base, UP, buff=0.2, aligned_edge=RIGHT)
            else:
                remitente.next_to(burbuja_base, UP, buff=0.1, aligned_edge=LEFT)
                
            return VGroup(remitente, burbuja_base)

        burbuja_pregunta = crear_burbuja(
            "¿Cuántas letras 'r' hay en 'strawberry'?", 
            color_fondo=MARRON_OSCURO, 
            color_texto=PAPEL_CREMA, es_usuario=True
        )
        
        burbuja_respuesta = crear_burbuja(
            "Hay 2 letras 'r' en 'strawberry'.", 
            color_fondo=FONDO_CAJA, 
            color_texto=TINTA_NEGRA, es_usuario=False,
            t2c_dict={"2": NARANJA_TERRACOTA}
        )

        grupo_chat = VGroup(burbuja_pregunta, burbuja_respuesta).arrange(DOWN, buff=0.5)
        burbuja_pregunta.shift(RIGHT * 1.5)
        burbuja_respuesta.shift(LEFT * 1.5)
        grupo_chat.next_to(linea, DOWN, buff=0.6)

        def crear_fresa():
            cuerpo = Polygon(
                [0, 0.3, 0], [-0.25, 0.1, 0], [-0.2, -0.2, 0], [0, -0.4, 0], [0.2, -0.2, 0], [0.25, 0.1, 0],
                fill_color="#E24A4A", fill_opacity=1, stroke_width=1.5, stroke_color=MARRON_OSCURO
            )
            hojas = Polygon(
                [0, 0.2, 0], [-0.2, 0.4, 0], [-0.1, 0.25, 0], [0, 0.45, 0], [0.1, 0.25, 0], [0.2, 0.4, 0],
                fill_color="#6B8E23", fill_opacity=1, stroke_width=1.5, stroke_color=MARRON_OSCURO
            )
            return VGroup(cuerpo, hojas).scale(0.85)

        fresa_der = crear_fresa().to_corner(DR).shift(UP*0.3 + LEFT*0.3)
        fresa_izq = crear_fresa().to_corner(DL).shift(UP*0.3 + RIGHT*0.3)

        self.add(fresa_der, fresa_izq) 
        
        self.play(FadeIn(burbuja_pregunta, shift=UP*0.2, scale=0.9))
        self.wait(0.5)
        
        self.play(FadeIn(burbuja_respuesta, shift=UP*0.2, scale=0.9))
        self.next_slide()

        texto_explicacion = Text(
            "Para el modelo, la palabra se divide en 'pedazos' (Tokens):", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        )
        
        token1 = self.crear_bloque("str", ancho=1.2)
        token2 = self.crear_bloque("aw", ancho=1.2)
        token3 = self.crear_bloque("berry", ancho=1.6)
        
        tokens_straw = VGroup(token1, token2, token3).arrange(RIGHT, buff=0.15)
        grupo_visual = VGroup(texto_explicacion, tokens_straw).arrange(DOWN, buff=0.5)
        grupo_visual.next_to(grupo_chat, DOWN, buff=1.0)

        self.play(FadeIn(texto_explicacion, shift=UP*0.2))
        self.play(
            LaggedStart(
                GrowFromCenter(token1),
                GrowFromCenter(token2),
                GrowFromCenter(token3),
                lag_ratio=0.2
            )
        )
        self.next_slide()

        cruz = Cross(tokens_straw, stroke_color=NARANJA_TERRACOTA, stroke_width=6)
        self.play(Create(cruz)) 
        self.next_slide()
        
        self.limpiar_pantalla()

    def diapo_tokenizacion(self):
        titulo, linea = self.crear_titulo(
            "La Tokenización", 
            palabra_clave="Tokenización", 
            color_clave=NARANJA_TERRACOTA
        )

        frase = "Confía en el tiempo, que suele dar dulces salidas a muchas amargas dificultades"

        p1 = Text(
            "1. Por palabra: Vocabulario infinito, muy ineficiente.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
            t2c={"Por palabra:": MARRON_OSCURO, "infinito": NARANJA_TERRACOTA}
        )
        tokens_palabra = frase.split(" ")
        
        ej_palabra = VGroup(*[
            self.crear_bloque(t, ancho=max(0.8, len(t) * 0.25)) 
            for t in tokens_palabra
        ]).arrange(RIGHT, buff=0.15)
        
        if ej_palabra.width > 12.5:
            ej_palabra.scale_to_fit_width(12.5)
            
        grupo_palabra = VGroup(p1, ej_palabra).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

        p2 = Text(
            "2. Por carácter: Secuencias larguísimas, pierde contexto.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
            t2c={"Por carácter:": MARRON_OSCURO, "pierde contexto": NARANJA_TERRACOTA}
        )
        tokens_caracter = [c if c != ' ' else '_' for c in frase]
        
        ej_caracter = VGroup(*[
            self.crear_bloque(t, ancho=0.3) 
            for t in tokens_caracter
        ]).arrange(RIGHT, buff=0.05)
        

        if ej_caracter.width > 12.5:
            ej_caracter.scale_to_fit_width(12.5)
            
        grupo_caracter = VGroup(p2, ej_caracter).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

 
        p3 = Text(
            "3. Sub-palabras (BPE): Algo más equilibrado.", 
            font=FUENTE, font_size=26, color=TINTA_NEGRA, weight=BOLD, 
            t2c={"Sub-palabras (BPE):": NARANJA_TERRACOTA, "equilibrado.": NARANJA_TERRACOTA}
        )
    
        tokens_bpe = [
            "Con", "f", "ía", "en", "el", "_tiem", "po", ",", "qu", "e", 
            "su", "ele", "d", "ar", "dul", "ces", "sali", "das", "a", 
            "mu", "chas", "amar", "gas", "di", "fic", "ul", "ta", "des"
        ]
        
        ej_bpe = VGroup(*[
            self.crear_bloque(t, ancho=max(0.8, len(t) * 0.25)) 
            for t in tokens_bpe
        ]).arrange(RIGHT, buff=0.15)
        
        if ej_bpe.width > 12.5:
            ej_bpe.scale_to_fit_width(12.5)
            
        grupo_bpe = VGroup(p3, ej_bpe).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

        puntos = VGroup(grupo_palabra, grupo_caracter, grupo_bpe).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        puntos.next_to(linea, DOWN, buff=0.5).shift(LEFT * 0.5)

        self.play(Write(titulo), Create(linea))
        self.next_slide()

        self.play(FadeIn(p1, shift=RIGHT * 0.3))
        self.play(LaggedStart(*[GrowFromCenter(b) for b in ej_palabra], lag_ratio=0.05))
        self.next_slide()

        self.play(FadeIn(p2, shift=RIGHT * 0.3))
        self.play(LaggedStart(*[GrowFromCenter(b) for b in ej_caracter], lag_ratio=0.01)) 
        self.next_slide()

        self.play(FadeIn(p3, shift=RIGHT * 0.3))
        self.play(LaggedStart(*[GrowFromCenter(b) for b in ej_bpe], lag_ratio=0.05))
        self.next_slide()

        self.limpiar_pantalla()

    def diapo_byte_pair_encoding(self):
        titulo, linea = self.crear_titulo(
            "Byte Pair Encoding (BPE)", 
            palabra_clave="Byte Pair Encoding (BPE)", 
            color_clave=NARANJA_TERRACOTA
        )

        explicacion_bpe = Text(
            "Fusión iterativa de los pares más frecuentes:", 
            font=FUENTE, font_size=28, color=MARRON_OSCURO
        ).next_to(linea, DOWN, buff=0.5)

        self.play(Write(titulo), Create(linea), FadeIn(explicacion_bpe))

        def calc_ancho(texto):
            return 0.6 + len(texto) * 0.3

        letras = ["t", "a", "c", "o", "t", "a", "c", "o"]

        fila_actual = VGroup(*[
            self.crear_bloque(letra, ancho=calc_ancho(letra)) 
            for letra in letras
        ]).arrange(RIGHT, buff=0.15).center().shift(UP*0.5)
        
        self.play(LaggedStart(*[FadeIn(b, shift=DOWN*0.5) for b in fila_actual], lag_ratio=0.1))
        self.next_slide()

        texto_paso1 = Text(
            "Paso 1: 't' y 'a' son el par más común", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        ).next_to(fila_actual, DOWN, buff=1)
        self.play(FadeIn(texto_paso1))
        
        fila_paso1 = VGroup(
            self.crear_bloque("ta", color_fondo=PAPEL_TAN, ancho=calc_ancho("ta")),
            self.crear_bloque("c", ancho=calc_ancho("c")),
            self.crear_bloque("o", ancho=calc_ancho("o")),
            self.crear_bloque("ta", color_fondo=PAPEL_TAN, ancho=calc_ancho("ta")),
            self.crear_bloque("c", ancho=calc_ancho("c")),
            self.crear_bloque("o", ancho=calc_ancho("o"))
        ).arrange(RIGHT, buff=0.15).center().shift(UP*0.5)

        self.play(
            ReplacementTransform(VGroup(fila_actual[0], fila_actual[1]), fila_paso1[0]),
            ReplacementTransform(fila_actual[2], fila_paso1[1]),
            ReplacementTransform(fila_actual[3], fila_paso1[2]),
            ReplacementTransform(VGroup(fila_actual[4], fila_actual[5]), fila_paso1[3]),
            ReplacementTransform(fila_actual[6], fila_paso1[4]),
            ReplacementTransform(fila_actual[7], fila_paso1[5]),
        )
        fila_actual = fila_paso1
        self.next_slide()

        texto_paso2 = Text(
            "Paso 2: 'c' y 'o' son el par más común", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        ).next_to(fila_actual, DOWN, buff=1)
        
        fila_paso2 = VGroup(
            self.crear_bloque("ta", color_fondo=PAPEL_TAN, ancho=calc_ancho("ta")),
            self.crear_bloque("co", color_fondo=CAJA_INFERIOR, ancho=calc_ancho("co")),
            self.crear_bloque("ta", color_fondo=PAPEL_TAN, ancho=calc_ancho("ta")),
            self.crear_bloque("co", color_fondo=CAJA_INFERIOR, ancho=calc_ancho("co"))
        ).arrange(RIGHT, buff=0.15).center().shift(UP*0.5)

        self.play(
            ReplacementTransform(texto_paso1, texto_paso2),
            ReplacementTransform(fila_actual[0], fila_paso2[0]),
            ReplacementTransform(VGroup(fila_actual[1], fila_actual[2]), fila_paso2[1]),
            ReplacementTransform(fila_actual[3], fila_paso2[2]),
            ReplacementTransform(VGroup(fila_actual[4], fila_actual[5]), fila_paso2[3]),
        )
        fila_actual = fila_paso2
        self.next_slide()

        texto_paso3 = Text(
            "Paso 3: 'ta' y 'co' forman un nuevo token", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        ).next_to(fila_actual, DOWN, buff=1)

        fila_paso3 = VGroup(
            self.crear_bloque("taco", color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=calc_ancho("taco")),
            self.crear_bloque("taco", color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=calc_ancho("taco"))
        ).arrange(RIGHT, buff=0.15).center().shift(UP*0.5)

        self.play(
            ReplacementTransform(texto_paso2, texto_paso3),
            ReplacementTransform(VGroup(fila_actual[0], fila_actual[1]), fila_paso3[0]),
            ReplacementTransform(VGroup(fila_actual[2], fila_actual[3]), fila_paso3[1]),
        )
        
        self.next_slide()
        self.limpiar_pantalla()
    
    def diapo_tamano_vocabulario(self):
        titulo, linea = self.crear_titulo(
            "El Tamaño del Vocabulario", 
            palabra_clave="Vocabulario", 
            color_clave=NARANJA_TERRACOTA
        )

        encabezados = VGroup(
            Text("Vocabulario", font=FUENTE, font_size=24, color=MARRON_OSCURO, weight=BOLD),
            Text("Total Tokens", font=FUENTE, font_size=24, color=MARRON_OSCURO, weight=BOLD),
            Text("Compresión", font=FUENTE, font_size=24, color=NARANJA_TERRACOTA, weight=BOLD)
        ).arrange(RIGHT, buff=1.5)

        linea_separadora = Line(LEFT, RIGHT, color=MARRON_OSCURO, stroke_width=2)
        
        def crear_fila(v, t, c, es_final=False):
            color_c = NARANJA_TERRACOTA if es_final else TINTA_NEGRA
            peso_c = BOLD if es_final else NORMAL
            return VGroup(
                Text(v, font=FUENTE, font_size=24, color=TINTA_NEGRA),
                Text(t, font=FUENTE, font_size=24, color=TINTA_NEGRA),
                Text(c, font=FUENTE, font_size=24, color=color_c, weight=peso_c)
            )

        fila1 = crear_fila("256", "2,168,312", "1.00x")
        fila2 = crear_fila("1,024", "724,453", "2.99x")
        fila3 = crear_fila("20,534", "460,900", "4.70x", es_final=True)

        for fila in [fila1, fila2, fila3]:
            for i in range(3):
                fila[i].set_x(encabezados[i].get_x())

        filas_datos = VGroup(fila1, fila2, fila3).arrange(DOWN, buff=0.4)
        
        tabla_interna = VGroup(encabezados, linea_separadora, filas_datos).arrange(DOWN, buff=0.3)
        linea_separadora.set_width(tabla_interna.width + 1)
        
        fondo_tabla = RoundedRectangle(
            width=tabla_interna.width + 1.5, 
            height=tabla_interna.height + 0.8, 
            corner_radius=0.15, 
            fill_color=FONDO_CAJA, 
            fill_opacity=1, 
            stroke_width=2,
            stroke_color=MARRON_OSCURO
        )
        
        grupo_tabla = VGroup(fondo_tabla, tabla_interna)
        grupo_tabla.next_to(linea, DOWN, buff=0.6)

        tradeoff_titulo = Text("El Trade-off (Compensación):", font=FUENTE, font_size=28, color=TINTA_NEGRA, weight=BOLD)
        
        pro_icon = Text("✅", font=FUENTE, font_size=24)
        pro_text = Text(
            "Más vocabulario = Textos cortos = Inferencia rápida", 
            font=FUENTE, font_size=24, color=MARRON_OSCURO, 
            t2c={"Inferencia rápida": NARANJA_TERRACOTA}
        )
        pro_group = VGroup(pro_icon, pro_text).arrange(RIGHT, buff=0.2)

        con_icon = Text("❌", font=FUENTE, font_size=24)
        con_text = Text(
            "Más vocabulario = Matriz gigante = Más VRAM", 
            font=FUENTE, font_size=24, color=MARRON_OSCURO, 
            t2c={"Más VRAM": NARANJA_TERRACOTA}
        )
        con_group = VGroup(con_icon, con_text).arrange(RIGHT, buff=0.2)
        
        textos_inferiores = VGroup(tradeoff_titulo, pro_group, con_group).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        textos_inferiores.next_to(grupo_tabla, DOWN, buff=0.6)

        self.play(Write(titulo), Create(linea))
        self.next_slide()

        self.play(FadeIn(fondo_tabla), FadeIn(encabezados), Create(linea_separadora))
        self.play(
            LaggedStart(
                FadeIn(fila1, shift=UP*0.2), 
                FadeIn(fila2, shift=UP*0.2), 
                FadeIn(fila3, shift=UP*0.2), 
                lag_ratio=0.3
            )
        )
        self.next_slide()
        
        self.play(FadeIn(tradeoff_titulo, shift=UP*0.2))
        self.play(FadeIn(pro_group, shift=RIGHT*0.2))
        self.play(FadeIn(con_group, shift=RIGHT*0.2))
        self.next_slide()
        
        self.limpiar_pantalla()

    def slide_que_es_un_tensor(self):
        titulo11, linea11 = self.crear_titulo("¿Qué es un Tensor?", palabra_clave="Tensor?", color_clave=NARANJA_TERRACOTA)
        
        colina_fondo = FunctionGraph(lambda x: 0.3 * math.sin(x / 1.5 + 2) - 2.8, color=NARANJA_TERRACOTA).set_stroke(opacity=0.15)
        colina_frente = FunctionGraph(lambda x: 0.5 * math.sin(x / 2) - 3.2, color=MARRON_OSCURO).set_stroke(opacity=0.25)
        llanura = VGroup(colina_fondo, colina_frente).set_z_index(-1)

        molino_adorno = crear_molino().scale(0.6).to_corner(DL, buff=0.8)
        sol_adorno = crear_sol_cervantino().scale(0.6).to_corner(UR, buff=0.8)
        tintero_adorno = crear_tintero_y_pluma().scale(0.6).to_corner(DR, buff=0.8)
        escudo_adorno = crear_escudo_y_lanza().scale(0.6).to_corner(UL, buff=0.8)
        
        sol_adorno.add_updater(lambda m, dt: m.rotate(dt * 0.15))

        self.play(
            Write(titulo11), 
            Create(linea11),
            FadeIn(llanura, shift=UP * 0.5, run_time=2),
            LaggedStart(
                FadeIn(sol_adorno, scale=0.5, shift=DOWN * 0.3),
                FadeIn(escudo_adorno, scale=0.5, shift=RIGHT * 0.3),
                FadeIn(molino_adorno, scale=0.5, shift=UP * 0.3),
                FadeIn(tintero_adorno, scale=0.5, shift=LEFT * 0.3),
                lag_ratio=0.2
            )
        )
        
        def_tensor = Text(
            "Un Tensor es un contenedor matemático\npara almacenar datos en múltiples dimensiones.", 
            font=FUENTE, font_size=32, 
            color=MARRON_OSCURO,
            t2c={"múltiples dimensiones.": NARANJA_TERRACOTA}
        ).next_to(linea11, DOWN, buff=0.4)
        
        self.play(Write(def_tensor))
        self.next_slide()
        
        txt_0d = Text("Escalar (0D) - Un solo valor", font=FUENTE, font_size=28, color=MARRON_OSCURO).next_to(def_tensor, DOWN, buff=0.8)
        escalar = self.crear_bloque("7", ancho=0.8, alto=0.8).next_to(txt_0d, DOWN, buff=0.5)
        
        self.play(Write(txt_0d))
        self.play(GrowFromCenter(escalar))
        self.next_slide()
        self.play(FadeOut(txt_0d), FadeOut(escalar))
        
        txt_1d = Text("Vector (1D) - Lista de valores", font=FUENTE, font_size=28, color=MARRON_OSCURO).next_to(def_tensor, DOWN, buff=0.8)
        vector = self.crear_matriz_bloques(1, 4, valores=["1", "5", "9", "2"]).next_to(txt_1d, DOWN, buff=0.5)
        
        self.play(Write(txt_1d))
        self.play(LaggedStart(*[FadeIn(b, shift=UP*0.2) for b in vector[0]], lag_ratio=0.1))
        self.next_slide()
        self.play(FadeOut(txt_1d), FadeOut(vector))
        
        txt_2d = Text("Matriz (2D) - Tabla de valores", font=FUENTE, font_size=28, color=MARRON_OSCURO).next_to(def_tensor, DOWN, buff=0.8)
        valores_matriz = ["3","1","4","2", "5","9","2","6", "5","3","5","8"]
        matriz = self.crear_matriz_bloques(3, 4, valores=valores_matriz).next_to(txt_2d, DOWN, buff=0.5)
        
        self.play(Write(txt_2d))
        bloques_anim = [FadeIn(b, shift=UP*0.2) for fila in matriz for b in fila]
        self.play(LaggedStart(*bloques_anim, lag_ratio=0.05))
        self.next_slide()
        self.play(FadeOut(txt_2d), FadeOut(matriz))
        
        txt_3d = Text("Tensor (3D+) - Cubo de valores", font=FUENTE, font_size=28, color=MARRON_OSCURO).next_to(def_tensor, DOWN, buff=0.5)
        
        matriz_base_3d = self.crear_matriz_bloques(3, 3, valores=["1","2","3","4","5","6","7","8","9"])
        for b in matriz_base_3d.submobjects:
            for sub_b in b.submobjects:
                sub_b[0].set_fill(opacity=0.3).set_stroke(opacity=0.3)

        matriz_medio_3d = self.crear_matriz_bloques(3, 3, color_fondo=PAPEL_TAN, valores=["9","8","7","6","5","4","3","2","1"]).shift(UP*0.25 + RIGHT*0.25)
        for b in matriz_medio_3d.submobjects:
            for sub_b in b.submobjects:
                sub_b[0].set_fill(opacity=0.6).set_stroke(opacity=0.6)

        matriz_top_3d = self.crear_matriz_bloques(3, 3, color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, valores=["2","4","6","8","0","2","4","6","8"]).shift(UP*0.5 + RIGHT*0.5)
        
        tensor_3d = VGroup(matriz_base_3d, matriz_medio_3d, matriz_top_3d).next_to(txt_3d, DOWN, buff=0.5).shift(LEFT * 0.25)
        
        self.play(Write(txt_3d))
        self.play(FadeIn(matriz_base_3d, shift=UP*0.3))
        self.play(FadeIn(matriz_medio_3d, shift=UP*0.3))
        self.play(FadeIn(matriz_top_3d, shift=UP*0.3))
        self.next_slide()
        
        self.play(FadeOut(txt_3d), FadeOut(tensor_3d), FadeOut(def_tensor))

        nota_ram = Text("Pero en memoria RAM, todos terminan siendo un arreglo plano...", font=FUENTE, font_size=26, color=NARANJA_TERRACOTA).next_to(linea11, DOWN, buff=0.5)
        self.play(Write(nota_ram))
        
        matriz_ram = self.crear_matriz_bloques(3, 4, valores=valores_matriz).next_to(nota_ram, DOWN, buff=0.8)
        self.play(FadeIn(matriz_ram))
        self.next_slide()

        bloques_individuales = [bloque for fila in matriz_ram for bloque in fila]
        grupo_plano = VGroup(*bloques_individuales)

        self.play(
            grupo_plano.animate.arrange(RIGHT, buff=0.05).scale(0.8).next_to(nota_ram, DOWN, buff=1.5),
            run_time=2
        )
        self.play(Indicate(grupo_plano, color=PAPEL_TAN))
        self.next_slide()

        sol_adorno.clear_updaters()
        self.limpiar_pantalla()

    def slide_softmax(self):
            titulo, linea = self.crear_titulo(
                "Softmax", 
                palabra_clave="Probabilidades", 
                color_clave=NARANJA_TERRACOTA
            )
            
            formula = MathTex(
                r"\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}", 
                color=TINTA_NEGRA, font_size=38
            ).next_to(linea, DOWN, buff=0.4)

            self.play(Write(titulo), Create(linea))
            self.play(Write(formula))
            self.next_slide()

            ancho_caja = 1.1
            
            col1 = VGroup(
                Text("1. Logits", font=FUENTE, font_size=18, color=MARRON_OSCURO, weight=BOLD),
                VGroup(*[self.crear_bloque(v, ancho=ancho_caja) for v in ["2.0", "1.0", "0.1"]]).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            flecha1 = Arrow(LEFT, RIGHT, color=TINTA_NEGRA).scale(0.5)
            txt_op1 = MathTex(r"\exp(x)", font_size=22, color=TINTA_NEGRA).next_to(flecha1, UP, buff=0.1)
            conector1 = VGroup(flecha1, txt_op1)

            col2 = VGroup(
                Text("2. Exp", font=FUENTE, font_size=18, color=MARRON_OSCURO, weight=BOLD),
                VGroup(*[self.crear_bloque(v, color_fondo=PAPEL_TAN, ancho=ancho_caja) for v in ["7.39", "2.72", "1.10"]]).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            flecha2 = Arrow(LEFT, RIGHT, color=TINTA_NEGRA).scale(0.5)
            txt_op2 = MathTex(r"\div \sum", font_size=22, color=TINTA_NEGRA).next_to(flecha2, UP, buff=0.1)
            conector2 = VGroup(flecha2, txt_op2)

            col3 = VGroup(
                Text("3. Prob", font=FUENTE, font_size=18, color=NARANJA_TERRACOTA, weight=BOLD),
                VGroup(*[self.crear_bloque(v, color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=ancho_caja) for v in ["66%", "24%", "10%"]]).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            acto1_horiz = VGroup(col1, conector1, col2, conector2, col3).arrange(RIGHT, buff=0.5).move_to(DOWN * 0.5)

            self.play(FadeIn(col1, shift=UP*0.2))
            self.play(Write(conector1), ReplacementTransform(col1[1].copy(), col2[1]), FadeIn(col2[0]))
            self.play(Write(conector2), ReplacementTransform(col2[1].copy(), col3[1]), FadeIn(col3[0]))
            self.next_slide()

            self.play(FadeOut(acto1_horiz), FadeOut(formula))

            titulo_error = Text("Problema: Un valor arruina el vector", font=FUENTE, font_size=32, color=NARANJA_TERRACOTA).move_to(titulo)
            self.play(ReplacementTransform(titulo, titulo_error))

            col_err_1 = VGroup(
                Text("Logits", font=FUENTE, font_size=18, color=MARRON_OSCURO, weight=BOLD),
                VGroup(
                    self.crear_bloque("800.0", color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=ancho_caja),
                    self.crear_bloque("2.0", ancho=ancho_caja),
                    self.crear_bloque("-1.0", ancho=ancho_caja)
                ).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            f_err = Arrow(LEFT, RIGHT, color=TINTA_NEGRA).scale(0.5)
            t_err = MathTex(r"\exp(x)", font_size=22, color=TINTA_NEGRA).next_to(f_err, UP, buff=0.1)
            conector_err = VGroup(f_err, t_err)

            col_err_2 = VGroup(
                Text("Exp", font=FUENTE, font_size=18, color=MARRON_OSCURO, weight=BOLD),
                VGroup(
                    self.crear_bloque("inf", color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=ancho_caja),
                    self.crear_bloque("7.39", color_fondo=PAPEL_TAN, ancho=ancho_caja),
                    self.crear_bloque("0.37", color_fondo=PAPEL_TAN, ancho=ancho_caja)
                ).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            flujo_error = VGroup(col_err_1, conector_err, col_err_2).arrange(RIGHT, buff=0.8).move_to(DOWN * 0.2)

            self.play(FadeIn(col_err_1))
            self.play(Flash(col_err_1[1][0], color=NARANJA_TERRACOTA))
            self.play(Write(conector_err))
            self.play(ReplacementTransform(col_err_1[1].copy(), col_err_2[1]), FadeIn(col_err_2[0]))
            self.play(Wiggle(col_err_2[1][0])) 
            
            nota_error = Text("Si intentamos dividir por 'inf', todo el vector se vuelve NaN.", font=FUENTE, font_size=20, color=MARRON_OSCURO).next_to(flujo_error, DOWN, buff=0.6)
            self.play(Write(nota_error))
            self.next_slide()

            self.play(FadeOut(flujo_error), FadeOut(nota_error))
            
            titulo_fix = Text("Solución: Restar el Máximo (Shift)", font=FUENTE, font_size=32, color=NARANJA_TERRACOTA).move_to(titulo_error)
            self.play(ReplacementTransform(titulo_error, titulo_fix))

            col_fix_1 = VGroup(
                Text("Logits", font=FUENTE, font_size=18, color=MARRON_OSCURO, weight=BOLD),
                VGroup(
                    self.crear_bloque("800.0", ancho=ancho_caja),
                    self.crear_bloque("2.0", ancho=ancho_caja),
                    self.crear_bloque("-1.0", ancho=ancho_caja)
                ).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            f_shift = Arrow(LEFT, RIGHT, color=TINTA_NEGRA).scale(0.5)
            t_shift = Text("- Max (x)", font=FUENTE, font_size=16, color=NARANJA_TERRACOTA, weight=BOLD).next_to(f_shift, UP, buff=0.1)
            conector_shift = VGroup(f_shift, t_shift)

            col_fix_2 = VGroup(
                Text("Shifted", font=FUENTE, font_size=18, color=NARANJA_TERRACOTA, weight=BOLD),
                VGroup(
                    self.crear_bloque("0.0", color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=ancho_caja), 
                    self.crear_bloque("-798.0", color_fondo=PAPEL_TAN, ancho=ancho_caja),
                    self.crear_bloque("-801.0", color_fondo=PAPEL_TAN, ancho=ancho_caja)
                ).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            f_exp2 = Arrow(LEFT, RIGHT, color=TINTA_NEGRA).scale(0.5)
            t_exp2 = MathTex(r"\exp(x)", font_size=22, color=TINTA_NEGRA).next_to(f_exp2, UP, buff=0.1)
            conector_exp2 = VGroup(f_exp2, t_exp2)

            col_fix_3 = VGroup(
                Text("Exp Seguro", font=FUENTE, font_size=18, color=MARRON_OSCURO, weight=BOLD),
                VGroup(
                    self.crear_bloque("1.0", color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=ancho_caja),
                    self.crear_bloque("0.0", ancho=ancho_caja), 
                    self.crear_bloque("0.0", ancho=ancho_caja)
                ).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            flujo_fix = VGroup(col_fix_1, conector_shift, col_fix_2, conector_exp2, col_fix_3).arrange(RIGHT, buff=0.35).move_to(DOWN * 0.2)

            self.play(FadeIn(col_fix_1))
            self.play(Write(conector_shift))
            self.play(ReplacementTransform(col_fix_1[1].copy(), col_fix_2[1]), FadeIn(col_fix_2[0]))
            self.play(Write(conector_exp2))
            self.play(ReplacementTransform(col_fix_2[1].copy(), col_fix_3[1]), FadeIn(col_fix_3[0]))
            
            nota_fix = Text(
                "Los valores gigantes se vuelven 0.0 y los demás se acercan a 0.\n¡El vector es matemáticamente idéntico pero 100% estable!", 
                font=FUENTE, font_size=18, color=TINTA_NEGRA
            ).next_to(flujo_fix, DOWN, buff=0.5)
            
            self.play(FadeIn(nota_fix, shift=UP*0.2))
            self.next_slide()

            self.limpiar_pantalla()

    def slide_broadcasting(self):
        titulo, linea = self.crear_titulo(
            "Broadcasting: Expansión Virtual", 
            palabra_clave="Expansión Virtual", 
            color_clave=PAPEL_TAN
        )

        val_base = ["1", "2", "3", "4", "5", "6"]
        val_vec = ["10", "20", "30"]
        val_res = ["11", "22", "33", "14", "25", "36"]

        matriz_base = self.crear_matriz_bloques(2, 3, valores=val_base)
        signo_mas = Text("+", font=FUENTE, font_size=40, color=TINTA_NEGRA)
        
        vector_real = VGroup(*[self.crear_bloque(val, color_fondo=PAPEL_TAN) for val in val_vec]).arrange(RIGHT, buff=0.05)
    
        ecuacion = VGroup(matriz_base, signo_mas, vector_real).arrange(RIGHT, buff=0.5).shift(LEFT * 2.5 + UP * 0.5)

        vector_real.align_to(matriz_base, UP)

        self.play(Write(titulo), Create(linea))
        self.play(
            FadeIn(matriz_base, shift=UP*0.2), 
            Write(signo_mas), 
            FadeIn(vector_real, shift=LEFT*0.2)
        )
        self.next_slide()

        vector_fantasma = vector_real.copy()
        
        self.play(vector_fantasma.animate.next_to(vector_real, DOWN, buff=0.05), run_time=1)
        
        self.play(
            *[b[0].animate.set_stroke(opacity=0.4).set_fill(opacity=0.2) for b in vector_fantasma],
            *[b[1].animate.set_opacity(0.4) for b in vector_fantasma],
            run_time=0.8
        )
        
        rectangulo_base = SurroundingRectangle(
            VGroup(vector_real, vector_fantasma), 
            color=MARRON_OSCURO, buff=0.1, corner_radius=0.1
        )
        caja_virtual = DashedVMobject(rectangulo_base, num_dashes=35)
        
        texto_virtual = Text("Matriz Virtual", font=FUENTE, font_size=16, color=MARRON_OSCURO).next_to(caja_virtual, DOWN, buff=0.15)

        self.play(Create(caja_virtual), Write(texto_virtual))
        self.next_slide()

        flecha_res = Arrow(LEFT, RIGHT, color=TINTA_NEGRA).next_to(caja_virtual, RIGHT, buff=0.5)

        matriz_res = self.crear_matriz_bloques(2, 3, color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, valores=val_res).next_to(flecha_res, RIGHT, buff=0.5)

        self.play(GrowArrow(flecha_res))
        
        self.play(
            ReplacementTransform(matriz_base.copy(), matriz_res),
            ReplacementTransform(VGroup(vector_real.copy(), vector_fantasma.copy()), matriz_res),
        )
        self.next_slide()
        
        def_broad = Text(
            "Se duplica el vector pequeño 'virtualmente' para coincidir\ncon la matriz grande, ahorrando muchísima memoria.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA, 
            t2c={"virtualmente": PAPEL_TAN, "ahorrando muchísima memoria": NARANJA_TERRACOTA}
        ).to_edge(DOWN, buff=1.0)
        
        self.play(FadeIn(def_broad, shift=UP*0.2))
        self.next_slide()
        self.limpiar_pantalla()

    def slide_strides(self):
        titulo, linea = self.crear_titulo(
            "Strides: Saltando en Memoria 1D", 
            palabra_clave="Strides:", 
            color_clave=NARANJA_TERRACOTA
        )

        camino_mancha = FunctionGraph(lambda x: 0.5 * math.sin(x) - 0.5, color=MARRON_OSCURO).set_opacity(0.15)
        camino_punteado = DashedVMobject(camino_mancha, num_dashes=45, dashed_ratio=0.5)
        
        lanza_fondo = Line(LEFT * 7 + DOWN * 2, RIGHT * 7 + UP * 2, color=NARANJA_TERRACOTA, stroke_width=2).set_opacity(0.15)
        
        decoracion_quijote = VGroup(camino_punteado, lanza_fondo).set_z_index(-2)

        molino_adorno = crear_molino().scale(0.6).to_corner(DL, buff=0.8)
        sol_adorno = crear_sol_cervantino().scale(0.6).to_corner(UR, buff=0.8)
        tintero_adorno = crear_tintero_y_pluma().scale(0.6).to_corner(DR, buff=0.8)
        escudo_adorno = crear_escudo_y_lanza().scale(0.6).to_corner(UL, buff=0.8)
        
        sol_adorno.add_updater(lambda m, dt: m.rotate(dt * 0.15))

        self.play(
            Write(titulo), 
            Create(linea),
            FadeIn(decoracion_quijote, run_time=2),
            LaggedStart(
                FadeIn(sol_adorno, scale=0.5, shift=DOWN * 0.3),
                FadeIn(escudo_adorno, scale=0.5, shift=RIGHT * 0.3),
                FadeIn(molino_adorno, scale=0.5, shift=UP * 0.3),
                FadeIn(tintero_adorno, scale=0.5, shift=LEFT * 0.3),
                lag_ratio=0.2
            )
        )

        arr_1d = VGroup(*[self.crear_bloque(str(i)) for i in range(6)])
        arr_1d.arrange(RIGHT, buff=0.1).shift(UP * 1.5)
        
        lbl_1d = Text("Memoria RAM (Física, 1D)", font=FUENTE, font_size=20, color=MARRON_OSCURO).next_to(arr_1d, UP, buff=0.3)

        self.play(FadeIn(arr_1d, shift=UP*0.2), FadeIn(lbl_1d, shift=UP*0.2))
        self.next_slide()

        fila1 = VGroup(*[arr_1d[i].copy() for i in range(3)]).arrange(RIGHT, buff=0.1)
        fila2 = VGroup(*[arr_1d[i].copy() for i in range(3, 6)]).arrange(RIGHT, buff=0.1)
        
        mat_shape = VGroup(fila1, fila2).arrange(DOWN, buff=0.1).shift(DOWN * 0.2)
        lbl_2d = Text("Shape Lógica (2x3)", font=FUENTE, font_size=20, color=MARRON_OSCURO).next_to(mat_shape, DOWN, buff=0.3)

        self.play(
            TransformFromCopy(VGroup(*arr_1d[0:3]), fila1),
            TransformFromCopy(VGroup(*arr_1d[3:6]), fila2),
            FadeIn(lbl_2d, shift=DOWN*0.2),
            run_time=1.5
        )
        self.next_slide()

        self.play(
            arr_1d[0][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
            arr_1d[3][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
            mat_shape[0][0][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
            mat_shape[1][0][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
        )

        arco_1d = CurvedArrow(arr_1d[0].get_top(), arr_1d[3].get_top(), angle=-PI/2, color=NARANJA_TERRACOTA)
        txt_stride_1d = Text("Stride = 3 pasos", font=FUENTE, font_size=20, color=NARANJA_TERRACOTA).next_to(arco_1d, UP, buff=0.1)
        
        arco_2d = CurvedArrow(mat_shape[0][0].get_left(), mat_shape[1][0].get_left(), angle=PI/2, color=NARANJA_TERRACOTA).shift(LEFT*0.1)
        txt_stride_2d = Text("+1 Fila", font=FUENTE, font_size=16, color=NARANJA_TERRACOTA).next_to(arco_2d, LEFT, buff=0.1)

        self.play(
            Create(arco_1d), Write(txt_stride_1d),
            Create(arco_2d), Write(txt_stride_2d)
        )
        self.next_slide()
        
        def_stride = Text(
            "Como todo es 1D en RAM, los 'strides' dictan\ncuántos casilleros avanzar para encontrar la siguiente fila.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA, 
            t2c={"1D en RAM": MARRON_OSCURO, "'strides'": NARANJA_TERRACOTA}
        ).to_edge(DOWN, buff=0.6) 
        
        self.play(FadeIn(def_stride, shift=UP*0.2))
        self.next_slide()
        
        sol_adorno.clear_updaters()
        self.limpiar_pantalla()

    def slide_masked_fill(self):
        titulo, linea = self.crear_titulo(
            "Masked Fill: Causalidad", 
            palabra_clave="Causalidad", 
            color_clave=NARANJA_TERRACOTA
        )

        val_mask = [
            "4.1", "1.2", "0.5", "2.1", 
            "3.3", "5.0", "1.8", "0.9", 
            "1.1", "2.4", "6.2", "1.5", 
            "0.8", "1.7", "3.0", "7.1"
        ]
        
        matriz_mask = self.crear_matriz_bloques(4, 4, valores=val_mask).scale(1.2).shift(DOWN * 0.2)
        
        lbl_matriz = Text("Scores de Atención (Previo al Masking)", font=FUENTE, font_size=20, color=MARRON_OSCURO).next_to(matriz_mask, UP, buff=0.3)
        
        self.play(Write(titulo), Create(linea))
        self.play(FadeIn(matriz_mask, shift=UP*0.2), FadeIn(lbl_matriz, shift=UP*0.2))
        self.next_slide()

        animaciones_mask = []
        animaciones_keep = []
        
        for i in range(4):
            for j in range(4):
                bloque = matriz_mask[i][j]
                if j > i: 
                    nuevo_bloque = self.crear_bloque("-∞", color_fondo=MARRON_OSCURO, color_texto=PAPEL_CREMA)
                    nuevo_bloque.match_height(bloque).move_to(bloque)
                    animaciones_mask.append(ReplacementTransform(bloque, nuevo_bloque))
                else:
                    animaciones_keep.append(
                        bloque[0].animate.set_stroke(color=PAPEL_TAN, width=3)
                    )

        lbl_mask = Text("Máscara Causal Aplicada (Masked Fill)", font=FUENTE, font_size=20, color=NARANJA_TERRACOTA).move_to(lbl_matriz)

        self.play(
            LaggedStart(*animaciones_mask, lag_ratio=0.15),
            *animaciones_keep,
            ReplacementTransform(lbl_matriz, lbl_mask),
            run_time=2
        )
        self.next_slide()

        def_mask = Text(
            "Se tapan los valores del 'futuro' con menos infinito (-∞).\nAl pasar por Softmax, esto se convierte en 0% de probabilidad.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA, 
            t2c={"-∞": NARANJA_TERRACOTA, "0% de probabilidad": NARANJA_TERRACOTA, "'futuro'": MARRON_OSCURO}
        ).to_edge(DOWN, buff=1.0)
        
        self.play(FadeIn(def_mask, shift=UP*0.2))
        self.next_slide()
        
        self.limpiar_pantalla()

    def slide_reshape_transpose(self):
        titulo, linea = self.crear_titulo(
            "Reshaping vs Transposing", 
            palabra_clave="vs", 
            color_clave=MARRON_OSCURO
        )
        
        titulo.set_color_by_t2c({"Reshaping": MARRON_OSCURO, "Transposing": NARANJA_TERRACOTA})
        
        linea_central = DashedLine(UP*2.2, DOWN*2.5, color=MARRON_OSCURO)

        self.play(Write(titulo), Create(linea), Create(linea_central))
        self.next_slide()

        txt_reshape = Text("Reshape", font=FUENTE, font_size=28, color=MARRON_OSCURO).move_to(LEFT * 3.5 + UP * 2.2)
        sub_reshape = Text("(Misma memoria)", font=FUENTE, font_size=16, color=TINTA_NEGRA).next_to(txt_reshape, DOWN, buff=0.1)
        
        valores_orig = ["1", "2", "3", "4", "5", "6"]
        
        mat_r_orig = self.crear_matriz_bloques(2, 3, valores=valores_orig).scale(0.8).next_to(sub_reshape, DOWN, buff=0.4)
        flecha_r = Arrow(mat_r_orig.get_bottom(), mat_r_orig.get_bottom() + DOWN * 0.8, color=MARRON_OSCURO)
        mat_r_final = self.crear_matriz_bloques(3, 2, color_fondo=PAPEL_TAN, valores=valores_orig).scale(0.8).next_to(flecha_r, DOWN, buff=0.2)

        self.play(Write(txt_reshape), Write(sub_reshape), FadeIn(mat_r_orig))
        self.play(GrowArrow(flecha_r))
        
        self.play(
            TransformFromCopy(mat_r_orig[0][0], mat_r_final[0][0]), # 1
            TransformFromCopy(mat_r_orig[0][1], mat_r_final[0][1]), # 2
            TransformFromCopy(mat_r_orig[0][2], mat_r_final[1][0]), # 3
            TransformFromCopy(mat_r_orig[1][0], mat_r_final[1][1]), # 4
            TransformFromCopy(mat_r_orig[1][1], mat_r_final[2][0]), # 5
            TransformFromCopy(mat_r_orig[1][2], mat_r_final[2][1]), # 6
            run_time=2
        )
        self.next_slide()

        txt_transpose = Text("Transpose", font=FUENTE, font_size=28, color=NARANJA_TERRACOTA).move_to(RIGHT * 3.5 + UP * 2.2)
        sub_transpose = Text("(Reorganización física)", font=FUENTE, font_size=16, color=TINTA_NEGRA).next_to(txt_transpose, DOWN, buff=0.1)

        mat_t_orig = self.crear_matriz_bloques(2, 3, valores=valores_orig).scale(0.8).next_to(sub_transpose, DOWN, buff=0.4)
        flecha_t = Arrow(mat_t_orig.get_bottom(), mat_t_orig.get_bottom() + DOWN * 0.8, color=NARANJA_TERRACOTA)
        
        valores_trans = ["1", "4", "2", "5", "3", "6"]
        mat_t_final = self.crear_matriz_bloques(3, 2, color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, valores=valores_trans).scale(0.8).next_to(flecha_t, DOWN, buff=0.2)

        self.play(Write(txt_transpose), Write(sub_transpose), FadeIn(mat_t_orig))
        self.play(GrowArrow(flecha_t))

        self.play(
            TransformFromCopy(mat_t_orig[0][0], mat_t_final[0][0]),                
            TransformFromCopy(mat_t_orig[0][1], mat_t_final[1][0], path_arc=PI/3),  
            TransformFromCopy(mat_t_orig[0][2], mat_t_final[2][0], path_arc=PI/3),  
            TransformFromCopy(mat_t_orig[1][0], mat_t_final[0][1], path_arc=-PI/3), 
            TransformFromCopy(mat_t_orig[1][1], mat_t_final[1][1], path_arc=-PI/3), 
            TransformFromCopy(mat_t_orig[1][2], mat_t_final[2][1]),                 
            run_time=2.5
        )
        self.next_slide()

        def_res_trans = Text(
            "Reshape solo cambia cómo leemos la lista.\nTranspose requiere copiar y cambiar el orden físico.", 
            font=FUENTE, font_size=22, color=TINTA_NEGRA, 
            t2c={"Reshape": MARRON_OSCURO, "Transpose": NARANJA_TERRACOTA}
        ).to_edge(DOWN, buff=0.5)
        
        self.play(FadeIn(def_res_trans, shift=UP*0.2))
        self.next_slide()
        
        self.limpiar_pantalla()

    def slide_forward_pass(self):
        titulo, linea = self.crear_titulo("Arquitectura: El Forward Pass", color_clave=NARANJA_TERRACOTA)
        
        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        
        llanuras_fondo = crear_llanuras_manchegas()
        
        self.play(
            Write(titulo), 
            Create(linea),
            FadeIn(sol_fondo),
            FadeIn(herradura_fondo, shift=UP*0.5),
            FadeIn(llanuras_fondo, shift=UP)
        )
        
        frase_completa = VGroup(
            Text('"En un lugar de la', font=FUENTE, font_size=42, color=TINTA_NEGRA),
            Text(' Mancha"', font=FUENTE, font_size=42, color=NARANJA_TERRACOTA)
        ).arrange(RIGHT, buff=0.1).move_to(UP * 1.8) 

        frase_base = frase_completa[0]
        posicion_destino_mancha = frase_completa[1].get_center()
        
        self.play(Write(frase_base))
        self.next_slide()
  
        BG_CAJA = PAPEL_CREMA
        BORDE_CAJA = MARRON_OSCURO
        TXT_PRINCIPAL = TINTA_NEGRA
        TXT_SECUNDARIO = MARRON_OSCURO 

        def crear_caja_nodo(label_sup, valor_array, label_inf, funcion_adorno=None, is_stack=False, highlight_last=False):
            grupo = VGroup()
            txt_sup = Text(label_sup, font="Monospace", font_size=16, weight=BOLD, color=TXT_SECUNDARIO)
            
            ANCHO = 2.5 
            ALTO = 1.6
            RADIO = 0.2
            
            sombra = RoundedRectangle(corner_radius=RADIO, width=ANCHO, height=ALTO)
            sombra.set_fill(MARRON_OSCURO, opacity=0.12).set_stroke(width=0)
            sombra.shift(RIGHT * 0.08 + DOWN * 0.08)
            
            caja = RoundedRectangle(corner_radius=RADIO, width=ANCHO, height=ALTO)
            caja.set_fill(color=BG_CAJA, opacity=1).set_stroke(color=BORDE_CAJA, width=2.5)
            
            color_array = NARANJA_TERRACOTA if highlight_last else TXT_PRINCIPAL
            txt_arr = Text(valor_array, font="Monospace", font_size=16, weight=BOLD, color=color_array) 
            txt_inf = Text(label_inf, font=FUENTE, font_size=15, color=TXT_PRINCIPAL) 
            
            if txt_arr.width > (ANCHO - 0.4):
                txt_arr.scale_to_fit_width(ANCHO - 0.4)
            if txt_inf.width > (ANCHO - 0.4):
                txt_inf.scale_to_fit_width(ANCHO - 0.4)
                
            Textos_caja = VGroup(txt_arr, txt_inf).arrange(DOWN, buff=0.25)
            
            if is_stack:
                caja_fondo1 = caja.copy().set_stroke(width=1.5, opacity=0.5).shift(RIGHT * 0.12 + UP * 0.12)
                caja_fondo2 = caja.copy().set_stroke(width=1.5, opacity=0.8).shift(RIGHT * 0.06 + UP * 0.06)
                fondo = VGroup(sombra, caja_fondo1, caja_fondo2, caja)
            else:
                fondo = VGroup(sombra, caja)
                
            caja_y_textos = VGroup(fondo, Textos_caja)
            Textos_caja.move_to(caja.get_center()) 
            
    
            if funcion_adorno:
                adorno = funcion_adorno().scale(0.65) 
                adorno.next_to(caja_y_textos, UP, buff=0.15)
                txt_sup.next_to(adorno, UP, buff=0.15)
                grupo.add(adorno)
                grupo.adorno = adorno 
            else:
                txt_sup.next_to(caja_y_textos, UP, buff=0.2)
                grupo.adorno = VGroup()
                
            grupo.add(txt_sup, caja_y_textos)
            grupo.caja_principal = caja 
            return grupo

        nodo_1 = crear_caja_nodo("1. Token_IDs", "[145, 892...]", "Tokens: En|un...", funcion_adorno=crear_tintero_y_pluma)
        nodo_2 = crear_caja_nodo("2. Embeddings", "[0.81, -0.2...]", "Vectores (768d)", funcion_adorno=crear_rueda_carreta)
        nodo_3 = crear_caja_nodo("3. Blocks", "[0.55, 0.9...]", "Atención (x12)", funcion_adorno=crear_molino, is_stack=True)
        nodo_4 = crear_caja_nodo("4. LayerNorm", "[0.12, -0.4...]", "Norm (768d)", funcion_adorno=crear_escudo_y_lanza)
        nodo_5 = crear_caja_nodo("5. LM_Head", "[..., 25.4...]", "Score Máximo", funcion_adorno=crear_pila_libros, highlight_last=True)

        pipeline = VGroup(nodo_1, nodo_2, nodo_3, nodo_4, nodo_5)
        pipeline.arrange(RIGHT, buff=0.4).scale(0.62).move_to(DOWN * 0.2)

        flechas = VGroup()
        for i in range(len(pipeline) - 1):
            flecha = Arrow(
                pipeline[i].caja_principal.get_right(), 
                pipeline[i+1].caja_principal.get_left(), 
                buff=0.1, color=MARRON_OSCURO, stroke_width=4, max_tip_length_to_length_ratio=0.2
            )
            flechas.add(flecha)


        self.play(
            AnimationGroup(*[FadeIn(nodo, shift=UP*0.3) for nodo in pipeline], lag_ratio=0.15),
            FadeIn(flechas, shift=UP*0.2),
            run_time=2.5
        )
        self.next_slide()

        tokens_viajeros = frase_base.copy()
        centro_primera_caja = pipeline[0].caja_principal.get_center()
        
        nucleo = Dot(color=WHITE, radius=0.08)
        aura = Dot(color=NARANJA_TERRACOTA, radius=0.2).set_opacity(0.6)
        punto_flujo = VGroup(aura, nucleo).move_to(centro_primera_caja)
        
        estela = TracedPath(punto_flujo.get_center, stroke_color=NARANJA_TERRACOTA, stroke_width=6, dissipating_time=0.4)
        
        panel_estado = RoundedRectangle(
            corner_radius=0.15, height=0.7, width=10, 
            fill_color=MARRON_OSCURO, fill_opacity=0.05, stroke_color=NARANJA_TERRACOTA, stroke_width=1
        )
        panel_estado.move_to(DOWN * 3.2)
        
        texto_estado = Text("Traduciendo a números...", font=FUENTE, font_size=24, color=NARANJA_TERRACOTA, weight=BOLD)
        texto_estado.move_to(panel_estado.get_center())

        self.play(
            ReplacementTransform(tokens_viajeros, punto_flujo, path_arc=-PI/4), 
            FadeIn(panel_estado, shift=UP*0.2),
            FadeIn(texto_estado, shift=UP*0.2),
            run_time=1.2
        )
        self.add(estela)
        
        self.play(
            pipeline[0].caja_principal.animate.scale(1.05).set_color(NARANJA_TERRACOTA),
            pipeline[0].adorno.animate.shift(UP*0.1), 
            rate_func=there_and_back, run_time=0.5
        )

        descripciones = [
            "Convirtiendo en vectores espaciales...",
            "Calculando atención y contexto (12 capas)...",
            "Estabilizando los datos...",
            "Calculando probabilidad de la siguiente palabra..."
        ]

        for i in range(len(flechas)):
            centro_siguiente_caja = pipeline[i+1].caja_principal.get_center()
            nuevo_texto_estado = Text(descripciones[i], font=FUENTE, font_size=24, color=NARANJA_TERRACOTA, weight=BOLD)
            nuevo_texto_estado.move_to(panel_estado.get_center())

            self.play(flechas[i].animate.set_color(NARANJA_TERRACOTA), run_time=0.2)
            

            self.play(
                punto_flujo.animate.move_to(centro_siguiente_caja), 
                ReplacementTransform(texto_estado, nuevo_texto_estado),
                run_time=0.8, 
                rate_func=smooth
            )
            texto_estado = nuevo_texto_estado

            self.play(
                flechas[i].animate.set_color(MARRON_OSCURO),
                pipeline[i+1].caja_principal.animate.scale(1.05).set_color(NARANJA_TERRACOTA), 
                pipeline[i+1].adorno.animate.shift(UP*0.1), 
                rate_func=there_and_back, 
                run_time=0.5
            )

        self.next_slide()

        prediccion_txt = Text('      Mancha"', font=FUENTE, font_size=42, color=NARANJA_TERRACOTA)
        prediccion_txt.move_to(pipeline[-1].caja_principal.get_center())
        
        self.play(
            FadeOut(estela),
            ReplacementTransform(punto_flujo, prediccion_txt), 
            FadeOut(panel_estado, shift=DOWN*0.2),
            FadeOut(texto_estado, shift=DOWN*0.2),
            run_time=0.8
        )
        self.next_slide()

        self.play(
            prediccion_txt.animate.move_to(posicion_destino_mancha),
            run_time=1.2,
            path_arc=-PI/4
        )
        
        frase_final = VGroup(frase_base, prediccion_txt)
        self.play(Flash(prediccion_txt, color=NARANJA_TERRACOTA, line_length=0.4, flash_radius=1.5, run_time=1))
        self.play(Circumscribe(frase_final, color=NARANJA_TERRACOTA, time_width=2, stroke_width=4))
        
        self.next_slide()
        self.limpiar_pantalla()

    def slide_embeddings(self):
        titulo_p1 = Text("Embeddings: ", font=FUENTE, font_size=42, weight=BOLD, color=TINTA_NEGRA)
        titulo_p2 = Text("De IDs a Vectores", font=FUENTE, font_size=42, weight=BOLD, color=NARANJA_TERRACOTA)
        titulo_completo = VGroup(titulo_p1, titulo_p2).arrange(RIGHT, buff=0.1)
        linea = Line(LEFT * 5, RIGHT * 5, color=MARRON_OSCURO).next_to(titulo_completo, DOWN)
        grupo_titulo = VGroup(titulo_completo, linea).to_edge(UP)
        
        self.play(Write(titulo_completo), Create(linea))

        ejes = Axes(
            x_range=[-4, 4, 1], y_range=[-3, 4, 1],
            x_length=8, y_length=5.0, 
            axis_config={"color": TINTA_NEGRA, "stroke_width": 2, "include_ticks": False}
        ).shift(DOWN * 0.1) 
        
        self.play(Create(ejes))

        v_feliz = np.array([2, -1.5, 0])
        v_triste = np.array([-2, -1.5, 0])
        v_caballero = np.array([2, 1.5, 0])
        v_quijote = np.array([-2, 1.5, 0])

        def crear_vector_2d(coord, color, label_text, direccion):
            p = ejes.c2p(*coord)
            flecha = Arrow(start=ejes.c2p(0,0), end=p, color=color, buff=0, stroke_width=5)
            lbl = Text(label_text, font=FUENTE, font_size=24, weight=BOLD, color=color)
            lbl.set_background_stroke(color=PAPEL_CREMA, width=4)
            lbl.next_to(p, direccion, buff=0.2)
            return VGroup(flecha, lbl)

        feliz = crear_vector_2d(v_feliz, MARRON_OSCURO, "Feliz", DR)
        triste = crear_vector_2d(v_triste, PAPEL_TAN, "Triste", DL)
        caballero = crear_vector_2d(v_caballero, MARRON_OSCURO, "Caballero", UR)
        quijote = crear_vector_2d(v_quijote, NARANJA_TERRACOTA, "Don Quijote", UL)

        self.play(Create(feliz), Create(triste))
        
        v_relacion = DashedLine(ejes.c2p(*v_feliz), ejes.c2p(*v_triste), color=CAJA_INFERIOR).set_stroke(width=4)
        v_relacion.add_tip()
        
        lbl_relacion = Text("- Felicidad", font=FUENTE, font_size=20, color=TINTA_NEGRA, weight=BOLD)
        lbl_relacion.set_background_stroke(color=PAPEL_CREMA, width=4)
        lbl_relacion.next_to(v_relacion, DOWN, buff=0.2)
        
        self.play(Create(v_relacion), Write(lbl_relacion))
        self.next_slide()

        self.play(Create(caballero))
        
        v_relacion_movida = v_relacion.copy().shift(ejes.c2p(*v_quijote) - ejes.c2p(*v_triste))
        lbl_relacion_movida = lbl_relacion.copy().next_to(v_relacion_movida, UP, buff=0.2)
        
        self.play(
            Transform(v_relacion, v_relacion_movida),
            Transform(lbl_relacion, lbl_relacion_movida)
        )
        
        punto_llegada = Dot(ejes.c2p(*v_quijote), color=NARANJA_TERRACOTA, radius=0.08)
        self.play(FadeIn(punto_llegada))
        
        self.play(ReplacementTransform(punto_llegada, quijote))
        self.play(Indicate(quijote, color=NARANJA_TERRACOTA))

        formula = MathTex(
            r"\vec{\text{Triste}} - \vec{\text{Feliz}} + \vec{\text{Caballero}} \approx \vec{\text{Don Quijote}}",
            font_size=38, color=TINTA_NEGRA
        ).to_edge(DOWN, buff=1.0) 
        
        self.play(Write(formula))
        self.next_slide()

        self.play(FadeOut(ejes, feliz, triste, caballero, quijote, v_relacion, lbl_relacion, formula))

        posicion_izquierda = LEFT * 4.5
        input_word = Text('"quijote"', font=FUENTE, font_size=40, color=TINTA_NEGRA, weight=BOLD).move_to(posicion_izquierda)
        
        id_token = Text("ID: 1605", font=FUENTE, font_size=36, weight=BOLD, color=FONDO_CAJA) 
        bg_id = RoundedRectangle(corner_radius=0.2, width=2.5, height=1, color=MARRON_OSCURO).set_fill(MARRON_OSCURO, 1)
        grupo_id = VGroup(bg_id, id_token).move_to(posicion_izquierda)

        self.play(FadeIn(input_word))
        self.next_slide()
        
        self.play(ReplacementTransform(input_word, grupo_id))
        self.next_slide()

        matriz_v = VGroup()
        filas, cols = 8, 10
        COLOR_VECTOR = NARANJA_TERRACOTA
        for i in range(filas):
            fila = VGroup()
            for j in range(cols):
                cuadro = RoundedRectangle(corner_radius=0.05, width=0.35, height=0.35).set_stroke(CAJA_INFERIOR, opacity=0.6)
                if i == 4:
                    cuadro.set_fill(COLOR_VECTOR, opacity=0.2).set_stroke(COLOR_VECTOR, opacity=0.8)
                fila.add(cuadro)
            matriz_v.add(fila.arrange(RIGHT, buff=0.08))
            
        matriz_v.arrange(DOWN, buff=0.08).move_to(RIGHT * 4)
        lbl_matriz = Text("Matriz de Embeddings", font=FUENTE, font_size=20, weight=BOLD, color=PAPEL_TAN).next_to(matriz_v, UP, buff=0.4)

        self.play(FadeIn(matriz_v), Write(lbl_matriz))
        self.next_slide()

        flecha_busqueda = Arrow(grupo_id.get_right(), matriz_v[4].get_left(), color=MARRON_OSCURO, buff=0.2)
        self.play(GrowArrow(flecha_busqueda))
        
        fila_sel = matriz_v[4].copy().set_fill(COLOR_VECTOR, 0.9).set_stroke(MARRON_OSCURO, 2)
        self.play(Transform(matriz_v[4], fila_sel))
        self.next_slide()

        vector_visual = VGroup()
        valores_ejemplo = ["0.1", "-0.4", "0.8", "-0.2", "0.5", "0.1", "-0.9", "0.3", "...", "0.7"]
        
        for val in valores_ejemplo:
            es_puntos = val == "..."
            bloque_v = RoundedRectangle(
                corner_radius=0.05, 
                width=0.7, height=0.7, 
                fill_color=COLOR_VECTOR if not es_puntos else FONDO_CAJA, 
                fill_opacity=0.8 if not es_puntos else 0
            ).set_stroke(MARRON_OSCURO if not es_puntos else FONDO_CAJA, 1.5 if not es_puntos else 0)
            
            txt_v = Text(val, font="Monospace", font_size=16, color=TINTA_NEGRA if not es_puntos else MARRON_OSCURO)
            txt_v.move_to(bloque_v.get_center())
            
            celda = VGroup(bloque_v, txt_v)
            vector_visual.add(celda)

        vector_visual.arrange(RIGHT, buff=0.1).move_to(DOWN * 2.5)
        
        lbl_final = Text("Vector Semántico", font=FUENTE, font_size=24, weight=BOLD, color=TINTA_NEGRA)
        lbl_size = Text("(768 dimensiones)", font=FUENTE, font_size=18, color=PAPEL_TAN)
        grupo_lbl_final = VGroup(lbl_final, lbl_size).arrange(DOWN, buff=0.1).next_to(vector_visual, UP, buff=0.3)

        self.play(
            ReplacementTransform(matriz_v[4].copy(), vector_visual),
            Write(grupo_lbl_final),
            run_time=1.5
        )
        
        self.play(
            vector_visual.animate.scale(1.05),
            rate_func=there_and_back,
            run_time=0.8
        )
        
        self.next_slide()
        self.limpiar_pantalla()

    def slide_position_embeddings(self):
        titulo_p1 = Text("Embeddings de ", font=FUENTE, font_size=42, weight=BOLD, color=TINTA_NEGRA)
        titulo_p2 = Text(" Posición", font=FUENTE, font_size=42, weight=BOLD, color=NARANJA_TERRACOTA)
        titulo_completo = VGroup(titulo_p1, titulo_p2).arrange(RIGHT, buff=0.1)
        linea = Line(LEFT * 5, RIGHT * 5, color=MARRON_OSCURO).next_to(titulo_completo, DOWN)
        grupo_titulo = VGroup(titulo_completo, linea).to_edge(UP)
        
        self.play(Write(titulo_completo), Create(linea))

        frase_l1 = Text("La virtud más es perseguida de los malos", font=FUENTE, font_size=32, color=TINTA_NEGRA)
        frase_l2 = Text("que amada de los buenos.", font=FUENTE, font_size=32, color=TINTA_NEGRA)
        frase = VGroup(frase_l1, frase_l2).arrange(DOWN, buff=0.15)
        
        nota_problema = Text("Los Transformers procesan todo en paralelo.\n¡Se pierde el orden de las palabras!", 
                            font=FUENTE, font_size=20, color=NARANJA_TERRACOTA).next_to(frase, DOWN, buff=0.5)
        
        self.play(FadeIn(frase))
        self.play(Write(nota_problema))
        self.next_slide()

        palabras = ["La", "virtud", "más", "es", "perseguida", "..."]
        tokens = VGroup(*[Text(w, font=FUENTE, font_size=28, color=TINTA_NEGRA) for w in palabras]).arrange(RIGHT, buff=0.4)
        
        posiciones = VGroup()
        for i, w in enumerate(palabras):
            texto_pos = f"Pos: {i}" if w != "..." else ""
            pos_token = Text(texto_pos, font=FUENTE, font_size=16, color=PAPEL_TAN)
            pos_token.next_to(tokens[i], DOWN, buff=0.2)
            posiciones.add(pos_token)

        grupo_tokens = VGroup(tokens, posiciones).move_to(UP * 1.5)
        
        self.play(
            FadeOut(nota_problema),
            ReplacementTransform(frase, grupo_tokens)
        )
        self.next_slide()

        def crear_vector_estilo(valores, color_principal):
            vector = VGroup()
            for val in valores:
                es_puntos = val == "..."
                bloque = RoundedRectangle(
                    corner_radius=0.05, width=0.8, height=0.6,
                    fill_color=color_principal if not es_puntos else FONDO_CAJA,
                    fill_opacity=0.8 if not es_puntos else 0
                ).set_stroke(MARRON_OSCURO if not es_puntos else FONDO_CAJA, 1.5)
                
                txt = Text(val, font="Monospace", font_size=16, color=TINTA_NEGRA)
                txt.move_to(bloque.get_center())
                vector.add(VGroup(bloque, txt))
            return vector.arrange(RIGHT, buff=0.1)

        self.play(
            *[FadeOut(t) for t in tokens[1:]],
            *[FadeOut(p) for p in posiciones[1:]],
            tokens[0].animate.move_to(LEFT * 4.5 + UP * 1.0),
            posiciones[0].animate.move_to(LEFT * 4.5 + DOWN * 0.2)
        )

        vec_token = crear_vector_estilo(["0.10", "-0.30", "0.20", "..."], PAPEL_TAN).shift(RIGHT * 1 + UP * 1.2)
        lbl_token = Text("Token Embedding (Significado)", font=FUENTE, font_size=16, weight=BOLD, color=MARRON_OSCURO).next_to(vec_token, UP, buff=0.2)
        
        vec_pos = crear_vector_estilo(["0.05", "-0.02", "0.01", "..."], NARANJA_TERRACOTA).shift(RIGHT * 1 + DOWN * 0.4)
        lbl_pos = Text("Position Embedding (Orden)", font=FUENTE, font_size=16, weight=BOLD, color=NARANJA_TERRACOTA).next_to(vec_pos, UP, buff=0.2)

        signo_mas = Text("+", font=FUENTE, font_size=36, weight=BOLD, color=TINTA_NEGRA).move_to(RIGHT * 1 + UP * 0.6)
        nota_suma = Text("Suma elemento a elemento (768 dimensiones)", font=FUENTE, font_size=16, color=MARRON_OSCURO).next_to(vec_pos, DOWN, buff=0.4)

        flecha_token = Arrow(tokens[0].get_right(), vec_token.get_left(), color=MARRON_OSCURO, buff=0.2)
        flecha_pos = Arrow(posiciones[0].get_right(), vec_pos.get_left(), color=NARANJA_TERRACOTA, buff=0.2)

        self.play(
            GrowArrow(flecha_token), FadeIn(vec_token), Write(lbl_token),
            GrowArrow(flecha_pos), FadeIn(vec_pos), Write(lbl_pos),
            Write(signo_mas), FadeIn(nota_suma)
        )
        self.next_slide()

        linea_suma = Line(LEFT * 1, RIGHT * 4.5, color=MARRON_OSCURO).next_to(nota_suma, DOWN, buff=0.2)
        vec_comb = crear_vector_estilo(["0.15", "-0.32", "0.21", "..."], CAJA_INFERIOR).next_to(linea_suma, DOWN, buff=0.6)
        lbl_comb = Text("Vector Combinado (Listo para el Transformer)", font=FUENTE, font_size=18, weight=BOLD, color=TINTA_NEGRA).next_to(vec_comb, UP, buff=0.2)

        self.play(Create(linea_suma))
        self.play(
            ReplacementTransform(vec_token.copy(), vec_comb),
            ReplacementTransform(vec_pos.copy(), vec_comb),
            Write(lbl_comb)
        )
        self.play(Indicate(vec_comb, color=NARANJA_TERRACOTA))
        self.next_slide()

        elementos_a_borrar = [vec_token, lbl_token, vec_pos, lbl_pos, signo_mas, nota_suma, linea_suma, vec_comb, lbl_comb, tokens[0], posiciones[0], flecha_token, flecha_pos]
        self.play(*[FadeOut(el) for el in elementos_a_borrar])

        titulo_matriz = Text("Tabla de Position Embeddings", font=FUENTE, font_size=28, weight=BOLD, color=TINTA_NEGRA).move_to(UP * 2)

        def crear_fila_matriz(pos_num, color):
            etiqueta = Text(f"Pos {pos_num}:", font=FUENTE, font_size=18, color=TINTA_NEGRA)
            cuadritos = VGroup(*[RoundedRectangle(corner_radius=0.02, width=0.4, height=0.3, fill_color=color, fill_opacity=0.5).set_stroke(MARRON_OSCURO, 1) for _ in range(8)])
            cuadritos.arrange(RIGHT, buff=0.05)
            puntos = Text("...", font_size=14, color=MARRON_OSCURO).next_to(cuadritos, RIGHT, buff=0.1)
            return VGroup(etiqueta, cuadritos, puntos).arrange(RIGHT, buff=0.3)

        f0 = crear_fila_matriz(0, PAPEL_TAN)
        f1 = crear_fila_matriz(1, PAPEL_TAN)
        puntos_v = Text("...", font_size=30, color=TINTA_NEGRA).rotate(PI/2)
        fn = crear_fila_matriz(1023, PAPEL_TAN)

        matriz_pos = VGroup(f0, f1, puntos_v, fn).arrange(DOWN, buff=0.3).next_to(titulo_matriz, DOWN, buff=0.5)

        llave = Brace(matriz_pos, direction=LEFT, color=MARRON_OSCURO)
        texto_llave = Text("1024 posiciones\n(block_size)", font=FUENTE, font_size=18, color=MARRON_OSCURO).next_to(llave, LEFT, buff=0.2)

        nota_context = Text("¡Este límite físico es el Context Window del modelo!", 
                            font=FUENTE, font_size=26, weight=BOLD, color=NARANJA_TERRACOTA).next_to(matriz_pos, DOWN, buff=0.6)

        self.play(Write(titulo_matriz))
        self.play(FadeIn(matriz_pos, shift=UP))
        self.play(GrowFromCenter(llave), Write(texto_llave))
        self.next_slide()

        self.play(Write(nota_context))
        self.play(Indicate(nota_context, color=NARANJA_TERRACOTA, scale_factor=1.1))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_layer_normalization(self):
        def crear_cajita(texto, bg_color, borde_color=MARRON_OSCURO, w=2.6, h=0.7, tam_fuente=20):
            caja = RoundedRectangle(corner_radius=0.1, width=w, height=h, 
                                    fill_color=bg_color, fill_opacity=1, 
                                    stroke_color=borde_color, stroke_width=2)
            lbl = Text(texto, font_size=tam_fuente, color=TINTA_NEGRA).move_to(caja.get_center())
            return VGroup(caja, lbl)

        def crear_vector_visual(numeros, bg_color, borde_color=MARRON_OSCURO):
            bloques = VGroup(*[
                crear_cajita(num, bg_color, borde_color, w=1.6, h=0.7, tam_fuente=18) 
                for num in numeros
            ]).arrange(RIGHT, buff=0.1)
            
            return bloques 

        titulo_p1 = Text("Layer ", font_size=42, weight=BOLD, color=TINTA_NEGRA)
        titulo_p2 = Text("Normalization", font_size=42, weight=BOLD, color=NARANJA_TERRACOTA) 
        titulo_completo = VGroup(titulo_p1, titulo_p2).arrange(RIGHT, buff=0.1)
        linea = Line(LEFT * 4, RIGHT * 4, color=MARRON_OSCURO).next_to(titulo_completo, DOWN)
        grupo_titulo = VGroup(titulo_completo, linea).to_edge(UP)
        
        self.play(Write(titulo_completo), Create(linea))
        self.next_slide()

        nota_caos = Text("Sin normalización: Los valores explotan o desaparecen", font_size=24, color=NARANJA_TERRACOTA).move_to(UP * 1.5)

        valores_inestables = ["8459.1", "-7302.4", "0.00001", "5120.9", "-9999.9", "..."]
        vec_inestable = crear_vector_visual(valores_inestables, bg_color="#F2D5CE", borde_color=NARANJA_TERRACOTA)
        vec_inestable.next_to(nota_caos, DOWN, buff=0.8).set_x(0) 
        
        self.play(Write(nota_caos))
        self.play(FadeIn(vec_inestable, shift=UP))
        self.play(Indicate(vec_inestable, color=NARANJA_TERRACOTA, scale_factor=1.1))
        self.next_slide()

        nota_estable = Text("Con LayerNorm: Se fuerza una Media=0 y Varianza=1", font_size=24, color=MARRON_OSCURO).move_to(UP * 1.5)
        
        valores_estables = ["1.34", "-1.15", "0.00", "0.89", "-1.52", "..."]
        vec_estable = crear_vector_visual(valores_estables, bg_color="#E8DCC4", borde_color=MARRON_OSCURO)
        vec_estable.next_to(nota_estable, DOWN, buff=0.8).set_x(0)
        
        self.play(
            ReplacementTransform(nota_caos, nota_estable),
            ReplacementTransform(vec_inestable, vec_estable)
        )
        self.next_slide()

        self.play(FadeOut(nota_estable), FadeOut(vec_estable))
        
        formula = MathTex(
            r"\text{output} = \frac{\text{input} - \mu}{\sqrt{\sigma^2 + \epsilon}} \times \gamma + \beta",
            substrings_to_isolate=[r"\epsilon", r"\times \gamma + \beta"],
            color=TINTA_NEGRA
        ).scale(1.2).move_to(UP * 0.5)

        lbl_formula = Text("La Fórmula (por cada token):", font_size=20, weight=BOLD, color=MARRON_OSCURO).next_to(formula, UP, buff=0.5)

        self.play(Write(lbl_formula), FadeIn(formula))
        self.next_slide()

        parte_eps = formula.get_part_by_tex(r"\epsilon")
        caja_eps = SurroundingRectangle(parte_eps, color=NARANJA_TERRACOTA, buff=0.05)
        nota_eps = Text("eps: Evita dividir por cero (ej. 0.00001)", font_size=18, color=NARANJA_TERRACOTA).next_to(caja_eps, DOWN, buff=0.5)
        
        self.play(Create(caja_eps), FadeIn(nota_eps, shift=UP))
        self.next_slide()

        self.play(FadeOut(caja_eps), FadeOut(nota_eps))
        
        parte_params = formula.get_part_by_tex(r"\times \gamma + \beta")
        caja_params = SurroundingRectangle(parte_params, color=MARRON_OSCURO, buff=0.1)
        nota_params = Text("gamma / beta: Parámetros aprendidos\n¡Le dan flexibilidad al modelo!", font_size=18, color=MARRON_OSCURO).next_to(caja_params, DOWN, buff=0.5)

        self.play(Create(caja_params), FadeIn(nota_params, shift=UP))
        self.next_slide()

        self.play(
            *[FadeOut(m) for m in [lbl_formula, formula, caja_params, nota_params]]
        )

        nota_final = Text("Se aplica 2 veces por capa:", font_size=28, weight=BOLD, color=TINTA_NEGRA)
        paso_1 = Text("1. Antes de Attention", font_size=24, color=MARRON_OSCURO)
        paso_2 = Text("2. Antes de FFN (MLP)", font_size=24, color=MARRON_OSCURO)
        
        textos_izq = VGroup(nota_final, paso_1, paso_2).arrange(DOWN, aligned_edge=LEFT, buff=0.4).to_edge(LEFT, buff=1).shift(UP * 0.5)

        b_in = crear_cajita("Input", "#E8DCC4")       
        b_ln1 = crear_cajita("LayerNorm 1", "#D9C8AA")  
        b_attn = crear_cajita("Attention", "#E6A87C", borde_color="#C0573E")
        b_ln2 = crear_cajita("LayerNorm 2", "#D9C8AA") 
        b_mlp = crear_cajita("MLP (FFN)", "#C2B280")    
        b_out = crear_cajita("Output", "#E8DCC4")       

        bloques = VGroup(b_in, b_ln1, b_attn, b_ln2, b_mlp, b_out).arrange(DOWN, buff=0.4)
        
        flechas = VGroup(*[
            Arrow(bloques[i].get_bottom(), bloques[i+1].get_top(), buff=0.1, 
                  max_tip_length_to_length_ratio=0.15, color=MARRON_OSCURO) 
            for i in range(len(bloques)-1)
        ])

        diagrama_simplificado = VGroup(bloques, flechas)
        diagrama_simplificado.scale(0.75).to_edge(RIGHT, buff=3.5).shift(DOWN * 0.2)

        self.play(Write(nota_final), FadeIn(diagrama_simplificado, shift=LEFT))
        self.next_slide()

        resalto_1 = SurroundingRectangle(b_ln1, color=NARANJA_TERRACOTA, stroke_width=4, buff=0.05)
        self.play(
            FadeIn(paso_1, shift=RIGHT), 
            Create(resalto_1), 
            b_ln1[0].animate.set_fill("#C0573E") 
        )
        self.next_slide()

        resalto_2 = SurroundingRectangle(b_ln2, color=NARANJA_TERRACOTA, stroke_width=4, buff=0.05)
        self.play(
            FadeIn(paso_2, shift=RIGHT), 
            Create(resalto_2), 
            b_ln2[0].animate.set_fill("#C0573E") 
        )
        self.next_slide()

        self.limpiar_pantalla()

    def slide_mha_acto1_intuicion(self):

        titulo, linea = self.crear_titulo(
            "Multi-Head Self-Attention",
            palabra_clave="Attention",
            color_clave=NARANJA_TERRACOTA
        )

        subtitulo = Text(
            "La Intuición (Q, K, V)",
            font=FUENTE, font_size=24, color=MARRON_OSCURO
        ).next_to(linea, DOWN, buff=0.5)

        self.play(Write(titulo), Create(linea), FadeIn(subtitulo, shift=DOWN))
        self.next_slide()

        nota_qkv = Text(
            "Para entender su contexto, cada palabra genera tres vectores:",
            font=FUENTE, font_size=24, color=MARRON_OSCURO
        ).next_to(subtitulo, DOWN, buff=0.8)

        self.play(FadeIn(nota_qkv, shift=UP))
        
        def crear_tarjeta(letra, nombre, pregunta, color):
            caja = RoundedRectangle(
                corner_radius=0.2, width=3, height=2.2,
                fill_color=FONDO_CAJA, fill_opacity=1,
                stroke_color=color, stroke_width=2
            )
            header_rect = Rectangle(
                width=3, height=0.7, fill_color=color, fill_opacity=1, stroke_width=0
            ).move_to(caja.get_top(), aligned_edge=UP)
            
            header = Intersection(header_rect, caja, color=color, fill_opacity=1, stroke_width=0)

            letra_txt = Text(letra, font=FUENTE, font_size=36, weight=BOLD, color=FONDO_CAJA).move_to(header.get_center())
            nombre_txt = Text(nombre, font=FUENTE, font_size=22, weight=BOLD, color=color).next_to(header, DOWN, buff=0.3)
            pregunta_txt = Text(pregunta, font=FUENTE, font_size=18, slant=ITALIC, color=TINTA_NEGRA).next_to(nombre_txt, DOWN, buff=0.15)

            return VGroup(caja, header, letra_txt, nombre_txt, pregunta_txt)

        tarjetas_qkv = VGroup(
            crear_tarjeta("Q", "Query", "¿Qué busco?", NARANJA_TERRACOTA),
            crear_tarjeta("K", "Key", "¿Qué ofrezco?", MARRON_OSCURO),
            crear_tarjeta("V", "Value", "Contenido", PAPEL_TAN)
        ).arrange(RIGHT, buff=0.6).next_to(nota_qkv, DOWN, buff=0.8)

        self.play(LaggedStart(*[FadeIn(t, shift=UP) for t in tarjetas_qkv], lag_ratio=0.2), run_time=1.5)
        self.next_slide()

        self.play(
            FadeOut(nota_qkv),
            FadeOut(tarjetas_qkv),
            FadeOut(subtitulo)
        )

        palabras = ["El", "hidalgo", "vio", "al", "gigante,", "pero", "este", "era", "un", "molino."]
        oracion = VGroup(*[Text(p, font=FUENTE, font_size=28, color=TINTA_NEGRA) for p in palabras]).arrange(RIGHT, buff=0.2)

        oracion.set_z_index(1)
        oracion.shift(DOWN * 1.0)

        nota_ejemplo = Text(
            "Ejemplo:",
            font=FUENTE, font_size=24, weight=BOLD, color=MARRON_OSCURO
        ).next_to(linea, DOWN, buff=0.8)

        self.play(Write(nota_ejemplo), FadeIn(oracion, shift=UP))
        self.next_slide()

        idx_query = 6        # "este"
        idx_key_fuerte = 4   # "gigante,"
        idx_key_debil = 1    # "hidalgo"

        caja_query = SurroundingRectangle(oracion[idx_query], color=NARANJA_TERRACOTA, corner_radius=0.1, buff=0.1)
        caja_query.set_z_index(1)
        label_q = Text("Query (busca antecedente)", font=FUENTE, font_size=16, color=NARANJA_TERRACOTA).next_to(caja_query, DOWN, buff=0.2)

        self.play(Create(caja_query), Write(label_q), oracion[idx_query].animate.set_color(NARANJA_TERRACOTA))
        self.next_slide()

        punto_inicio = oracion[idx_query].get_center() + UP * 0.4
        punto_fin_fuerte = oracion[idx_key_fuerte].get_center() + UP * 0.4
        punto_fin_debil = oracion[idx_key_debil].get_center() + UP * 0.4

        label_k_fuerte = Text("Key", font=FUENTE, font_size=14, color=NARANJA_TERRACOTA).next_to(oracion[idx_key_fuerte], UP, buff=0.1)
        label_k_debil = Text("Key", font=FUENTE, font_size=14, color=MARRON_OSCURO).next_to(oracion[idx_key_debil], UP, buff=0.1).set_opacity(0.5)

        flecha_fuerte = CurvedArrow(
            start_point=punto_inicio, end_point=punto_fin_fuerte,
            angle=PI/1.2, color=NARANJA_TERRACOTA, stroke_width=3
        ).set_z_index(-1) 
        
        peso_fuerte = Text("0.85", font=FUENTE, font_size=22, color=NARANJA_TERRACOTA).next_to(flecha_fuerte.point_from_proportion(0.5), UP, buff=0.1).set_z_index(-1)

        flecha_debil = CurvedArrow(
            start_point=punto_inicio, end_point=punto_fin_debil,
            angle=PI/2, color=MARRON_OSCURO, stroke_width=3
        ).set_z_index(-1)
        
        flecha_debil.set_stroke(opacity=0.4)
        flecha_debil.get_tip().set_fill(opacity=0.4)
        flecha_debil.get_tip().set_stroke(opacity=0.4)
        
        peso_debil = Text("0.10", font=FUENTE, font_size=20, color=MARRON_OSCURO).next_to(flecha_debil.point_from_proportion(0.5), UP, buff=0.1).set_opacity(0.4).set_z_index(-1)

        self.play(
            FadeIn(label_k_debil, shift=DOWN*0.2), oracion[idx_key_debil].animate.set_opacity(0.5),
            FadeIn(label_k_fuerte, shift=DOWN*0.2), oracion[idx_key_fuerte].animate.set_color(NARANJA_TERRACOTA),
            run_time=1
        )
        
        self.play(
            Create(flecha_debil), 
            Create(flecha_fuerte),
            run_time=1.2
        )
        
        self.play(
            FadeIn(peso_debil, shift=DOWN*0.1),
            FadeIn(peso_fuerte, shift=DOWN*0.1),
            run_time=0.8
        )
        self.next_slide()

        nota_final = Text(
            "Value: 'este' absorbe el significado de 'gigante'",
            font=FUENTE, font_size=22, color=MARRON_OSCURO
        ).next_to(label_q, DOWN, buff=0.8)
        
        self.play(Write(nota_final))

        particulas = VGroup(*[
            Dot(point=oracion[idx_key_fuerte].get_center() + np.random.uniform(-0.2, 0.2, 3), radius=0.04, color=NARANJA_TERRACOTA).set_z_index(2)
            for _ in range(20)
        ])

        self.play(FadeIn(particulas, lag_ratio=0.1), run_time=0.5)
        self.play(
            LaggedStart(*[p.animate.move_to(oracion[idx_query].get_center() + np.random.uniform(-0.1, 0.1, 3)) for p in particulas], lag_ratio=0.03),
            oracion[idx_query].animate.scale(1.15).set_color(NARANJA_TERRACOTA),
            run_time=1.5
        )
        self.play(
            FadeOut(particulas, shift=DOWN*0.2), 
            oracion[idx_query].animate.scale(1/1.15)
        )
        self.next_slide()

        elementos_escena = VGroup(
            oracion, caja_query, label_q, flecha_fuerte, flecha_debil,
            peso_fuerte, peso_debil, nota_ejemplo, nota_final, label_k_fuerte, label_k_debil
        )
        self.play(FadeOut(elementos_escena))
        
    def slide_mha_acto2_formula(self):
        titulo, linea = self.crear_titulo("Multi-Head Self-Attention", palabra_clave="Attention", color_clave=NARANJA_TERRACOTA)
        subtitulo = Text("La Ecuación de Atención", font=FUENTE, font_size=24, color=MARRON_OSCURO).next_to(linea, DOWN)
        
        self.play(Write(titulo), Create(linea), FadeIn(subtitulo, shift=DOWN))
        self.next_slide()
        
        formula = MathTex(
            r"\text{Attention}(Q, K, V) = ", # Índice 0
            r"\text{softmax}",               # Índice 1
            r"\left( \frac{",                # Índice 2
            r"Q K^T",                        # Índice 3
            r"}{",                           # Índice 4
            r"\sqrt{d_k}",                   # Índice 5
            r"} \right) ",                   # Índice 6
            r"V",                            # Índice 7
            color=TINTA_NEGRA, font_size=48
        ).move_to(UP * 0.5)

        self.play(FadeIn(formula, shift=UP))
        self.next_slide()

        partes_explicacion = [
            (3, "Similitud: Medimos qué tanto se relacionan los tokens.", NARANJA_TERRACOTA),
            (5, "Escalado: Estabilizamos gradientes.", MARRON_OSCURO),
            (1, "Normalización: Convertimos scores a probabilidades.", TINTA_NEGRA),
            (7, "Contexto: Extraemos la información ponderada.", PAPEL_TAN)
        ]

        caja_enfoque = None
        txt_enfoque = None

        for idx, desc, col in partes_explicacion:
            parte_formula = formula[idx]
            nueva_caja = SurroundingRectangle(parte_formula, color=col, buff=0.1)
            
            nuevo_txt = Text(desc, font=FUENTE, font_size=22, color=col).next_to(formula, DOWN, buff=1.5)
            
            if caja_enfoque:
                self.play(ReplacementTransform(caja_enfoque, nueva_caja), ReplacementTransform(txt_enfoque, nuevo_txt))
            else:
                self.play(Create(nueva_caja), FadeIn(nuevo_txt, shift=UP))
            
            caja_enfoque, txt_enfoque = nueva_caja, nuevo_txt
            self.next_slide()

        self.limpiar_pantalla()

    def slide_mha_acto3_calculo(self):

        titulo, linea = self.crear_titulo("Flujo Cálculo Self-Attention", palabra_clave="Flujo", color_clave=NARANJA_TERRACOTA)
        self.play(Write(titulo), Create(linea))
        self.next_slide()

        def crear_caja(texto, bg_color, ancho=1.6, alto=1.2, font_size=24, es_vertical=False):
            w = ancho if not es_vertical else alto
            h = alto if not es_vertical else ancho
            
            caja = RoundedRectangle(
                corner_radius=0.15, width=w, height=h,
                fill_color=bg_color, fill_opacity=1,
                stroke_color=TINTA_NEGRA, stroke_width=2
            )

            if '\\' in texto or '^' in texto:
                lbl = MathTex(texto, color=TINTA_NEGRA, font_size=font_size+8)
            else:
                lbl = Text(texto, font_size=font_size, color=TINTA_NEGRA)
            
            lbl.move_to(caja.get_center())
            return VGroup(caja, lbl)

        col_1_x = -6.0  
        col_2_x = -4.2 
        col_3_x = -2.4  
        col_4_x = -0.4 
        col_5_x = 1.4   
        col_6_x = 3.2   
        col_7_x = 5.2  
        col_8_x = 6.5  

        fila_q_y = 1.1
        fila_k_y = 0.0
        fila_v_y = -1.1

        X = MathTex("X", color=TINTA_NEGRA, font_size=40).move_to([col_1_x, 0, 0])
        nodo_x = Dot(radius=0.06, color=TINTA_NEGRA).next_to(X, RIGHT, buff=0.15)
        linea_x = Line(X.get_right(), nodo_x.get_center(), color=TINTA_NEGRA, stroke_width=3)

        color_w = "#C4C4FF" 
        W_q = crear_caja(r"W^{(q)}", color_w, ancho=1.4, alto=0.8).move_to([col_2_x, fila_q_y, 0])
        W_k = crear_caja(r"W^{(k)}", color_w, ancho=1.4, alto=0.8).move_to([col_2_x, fila_k_y, 0])
        W_v = crear_caja(r"W^{(v)}", color_w, ancho=1.4, alto=0.8).move_to([col_2_x, fila_v_y, 0])

        def ruta_angulada_l2r(start, end_mobj):
            p_mid = [start[0], end_mobj.get_y(), 0]
            l1 = Line(start, p_mid, color=TINTA_NEGRA, stroke_width=3)
            l2 = Arrow(p_mid, end_mobj.get_left(), buff=0.05, color=TINTA_NEGRA, stroke_width=3)
            return VGroup(l1, l2)

        ruta_q = ruta_angulada_l2r(nodo_x.get_center(), W_q)
        ruta_k = Arrow(nodo_x.get_center(), W_k.get_left(), buff=0.05, color=TINTA_NEGRA, stroke_width=3)
        ruta_v = ruta_angulada_l2r(nodo_x.get_center(), W_v)


        Q = MathTex("Q", color=TINTA_NEGRA, font_size=36).move_to([col_3_x, fila_q_y, 0])
        K = MathTex("K", color=TINTA_NEGRA, font_size=36).move_to([col_3_x, fila_k_y, 0])
        V = MathTex("V", color=TINTA_NEGRA, font_size=36).move_to([col_3_x, fila_v_y, 0])

        a_wq = Arrow(W_q.get_right(), Q.get_left(), buff=0.1, color=TINTA_NEGRA, stroke_width=3)
        a_wk = Arrow(W_k.get_right(), K.get_left(), buff=0.1, color=TINTA_NEGRA, stroke_width=3)
        a_wv = Arrow(W_v.get_right(), V.get_left(), buff=0.1, color=TINTA_NEGRA, stroke_width=3)

        y_qk_branch = (fila_q_y + fila_k_y) / 2
        matmul_1 = crear_caja("mat mul", "#FFCC99", ancho=1.2, alto=1.8).move_to([col_4_x, y_qk_branch, 0])
  
        a_q_mm = Arrow(Q.get_right(), [matmul_1.get_left()[0], Q.get_y(), 0], buff=0.05, color=TINTA_NEGRA, stroke_width=3)
        a_k_mm = Arrow(K.get_right(), [matmul_1.get_left()[0], K.get_y(), 0], buff=0.05, color=TINTA_NEGRA, stroke_width=3)

        scale_box = crear_caja("scale", "#FFFFCC", ancho=1.2, alto=1.8).move_to([col_5_x, y_qk_branch, 0])
        a_mm_sc = Arrow(matmul_1.get_right(), scale_box.get_left(), buff=0.05, color=TINTA_NEGRA, stroke_width=3)

        softmax_box = crear_caja("softmax", "#CCFFCC", ancho=1.2, alto=1.8).move_to([col_6_x, y_qk_branch, 0])
        a_sc_sm = Arrow(scale_box.get_right(), softmax_box.get_left(), buff=0.05, color=TINTA_NEGRA, stroke_width=3)

        matmul_2 = crear_caja("mat mul", "#FFCC99", ancho=1.8, alto=2.8).move_to([col_7_x, 0, 0])
        
        a_sm_mm2 = Arrow(softmax_box.get_right(), [matmul_2.get_left()[0], softmax_box.get_y(), 0], buff=0.05, color=TINTA_NEGRA, stroke_width=3)
        a_v_mm2 = Arrow(V.get_right(), [matmul_2.get_left()[0], V.get_y(), 0], buff=0.05, color=TINTA_NEGRA, stroke_width=3)

        Y = MathTex("Y", color=TINTA_NEGRA, font_size=40).move_to([col_8_x, 0, 0])
        a_mm2_y = Arrow(matmul_2.get_right(), Y.get_left(), buff=0.05, color=TINTA_NEGRA, stroke_width=3)

        zona_texto = DOWN * 2.8
        
        panel_fondo = RoundedRectangle(
            corner_radius=0.2, width=12.5, height=0.8, 
            fill_color="#F4EBD0", fill_opacity=1, 
            stroke_color=TINTA_NEGRA, stroke_width=2
        ).move_to(zona_texto)

        txt_1 = Text("1. Proyección: La entrada X se divide en matrices Query, Key y Value.", font_size=22, color=TINTA_NEGRA).move_to(panel_fondo.get_center())
        txt_2 = Text("2. Similitud: Se multiplican Q y K para ver qué tokens importan más.", font_size=22, color=TINTA_NEGRA).move_to(panel_fondo.get_center())
        txt_3 = Text("3. Estabilización: Se escala y aplica Softmax para obtener porcentajes (0 a 1).", font_size=22, color=TINTA_NEGRA).move_to(panel_fondo.get_center())
        txt_4 = Text("4. Contexto Final: Se filtran los Values (V) según esos porcentajes.", font_size=22, color=TINTA_NEGRA).move_to(panel_fondo.get_center())

        self.play(FadeIn(panel_fondo, shift=UP), Write(txt_1)) 
        self.play(Write(X), Create(linea_x), FadeIn(nodo_x))
        
        self.play(AnimationGroup(Create(ruta_q), Create(ruta_k), Create(ruta_v), lag_ratio=0.2))
        self.play(AnimationGroup(GrowFromCenter(W_q), GrowFromCenter(W_k), GrowFromCenter(W_v), lag_ratio=0.1))
        
        self.play(
            AnimationGroup(Create(a_wq), Create(a_wk), Create(a_wv), lag_ratio=0.1),
            AnimationGroup(FadeIn(Q, shift=RIGHT), FadeIn(K, shift=RIGHT), FadeIn(V, shift=RIGHT), lag_ratio=0.1)
        )
        self.next_slide()

        self.play(FadeTransform(txt_1, txt_2))
        self.play(Create(a_q_mm), Create(a_k_mm))
        self.play(GrowFromCenter(matmul_1))
        self.next_slide()

        self.play(FadeTransform(txt_2, txt_3))
        self.play(Create(a_mm_sc))
        self.play(GrowFromCenter(scale_box))
        self.play(Create(a_sc_sm))
        self.play(GrowFromCenter(softmax_box))
        self.next_slide()

        self.play(FadeTransform(txt_3, txt_4))
        self.play(Create(a_sm_mm2), Create(a_v_mm2)) 
        self.play(GrowFromCenter(matmul_2))
        self.next_slide()

        self.play(Create(a_mm2_y))
        self.play(FadeIn(Y, shift=RIGHT), Flash(Y, color=NARANJA_TERRACOTA, line_length=0.3))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_mha_acto4_multihead(self):
        titulo, linea = self.crear_titulo("Multi-Head Self-Attention", palabra_clave="Attention", color_clave=NARANJA_TERRACOTA)
        subtitulo = Text("¿Por qué 'Multi-Head'?", font=FUENTE, font_size=24, color=MARRON_OSCURO).next_to(linea, DOWN)
        
        self.play(Write(titulo), Create(linea), FadeIn(subtitulo, shift=DOWN))
        self.next_slide()

        vector_completo = Rectangle(width=10, height=0.8, fill_color=FONDO_CAJA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2)
        txt_dim = Text("Vector de Incrustación (ej. 768 dimensiones)", font=FUENTE, font_size=20, color=TINTA_NEGRA).move_to(vector_completo)
        
        self.play(FadeIn(vector_completo), FadeIn(txt_dim))
        self.next_slide()

        colores_h = [NARANJA_TERRACOTA, MARRON_OSCURO, PAPEL_TAN, CAJA_INFERIOR] * 3 
        
        cabezas = VGroup(*[
            Rectangle(width=10/12, height=1.2, fill_color=colores_h[i], fill_opacity=0.9, stroke_color=FONDO_CAJA, stroke_width=1)
            for i in range(12)
        ]).arrange(RIGHT, buff=0.02).move_to(vector_completo)

        txt_mh = Text("12 Cabezas de Atención (64 dims c/u)", 
                      font=FUENTE, font_size=22, color=MARRON_OSCURO).next_to(cabezas, DOWN, buff=1)

        self.play(
            ReplacementTransform(vector_completo, cabezas),
            FadeOut(txt_dim),
            FadeIn(txt_mh, shift=UP)
        )
        
        self.play(LaggedStart(*[Indicate(c, scale_factor=1.1, color=PAPEL_CREMA) for c in cabezas], lag_ratio=0.1))
        self.next_slide()

        final_msg = Text("Cada cabeza hace su propia multiplicación de matrices Q, K, V", 
                         font=FUENTE, font_size=20, color=TINTA_NEGRA).next_to(txt_mh, DOWN, buff=0.5)
        self.play(Write(final_msg))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_causal_masking(self):
        titulo, linea = self.crear_titulo("Causal Masking", palabra_clave="Masking", color_clave=NARANJA_TERRACOTA)
        self.play(Write(titulo), Create(linea))
        
        subtitulo = Text("El 'Triángulo de No Mirar' (No Peeking)", font=FUENTE, font_size=26, color=MARRON_OSCURO)
        
        tokens = ["El", "amor", "nunca", "hizo", "ningún", "cobarde"]
        frase_display = VGroup(
            Text("Secuencia: ", font=FUENTE, font_size=24, color=TINTA_NEGRA, weight=BOLD),
            Text(" ".join([f"[{t}]" for t in tokens]), font=FUENTE, font_size=24, color=NARANJA_TERRACOTA)
        ).arrange(RIGHT)

        grupo_intro = VGroup(subtitulo, frase_display).arrange(DOWN, buff=0.5).next_to(linea, DOWN, buff=0.6)

        self.play(FadeIn(grupo_intro, shift=UP))
        self.next_slide()
        self.play(FadeOut(grupo_intro))

        texto_explicativo = Text("Sin máscara: 'El' ya sabe que la frase termina en 'cobarde'", 
                                 font=FUENTE, font_size=20, weight=BOLD, color=NARANJA_TERRACOTA).next_to(linea, DOWN, buff=0.4)
        self.play(FadeIn(texto_explicativo))

        val_scores = [f"{random.uniform(0.1, 3.5):.1f}" for _ in range(36)]
        mat_scores = self.crear_matriz_bloques(6, 6, color_fondo=FONDO_CAJA, valores=val_scores)
        
        etiquetas_filas = VGroup()
        for i, t in enumerate(tokens):
            etiqueta = Text(t, font=FUENTE, font_size=14, color=MARRON_OSCURO)
            etiqueta.next_to(mat_scores[i][0], LEFT, buff=0.25)
            etiquetas_filas.add(etiqueta)

        etiquetas_cols = VGroup()
        for j, t in enumerate(tokens):
            etiqueta = Text(t, font=FUENTE, font_size=14, color=MARRON_OSCURO).rotate(PI/4)
            etiqueta.next_to(mat_scores[0][j], UP, buff=0.15)
            etiquetas_cols.add(etiqueta)

        grupo_atencion = VGroup(mat_scores, etiquetas_filas, etiquetas_cols).scale(0.85).move_to(DOWN * 0.6)
        
        self.play(FadeIn(grupo_atencion, shift=UP))
        self.next_slide()

        animaciones_trampa = []
        for i in range(6):
            for j in range(6):
                if j > i: 
                    bloque = mat_scores[i][j]
                    animaciones_trampa.append(bloque[0].animate.set_fill(NARANJA_TERRACOTA, opacity=0.3))
                    animaciones_trampa.append(bloque[0].animate.set_stroke(NARANJA_TERRACOTA, width=2))

        self.play(*animaciones_trampa, run_time=1.2)
        self.next_slide()

        texto_infinito = Text("Solución: Forzar el valor -Infinito en el triángulo superior", 
                              font=FUENTE, font_size=20, weight=BOLD, color=MARRON_OSCURO).move_to(texto_explicativo)
        self.play(Transform(texto_explicativo, texto_infinito))

        animaciones_infinito = []
        textos_a_reemplazar = []
        
        for i in range(6):
            for j in range(6):
                if j > i:
                    bloque_score = mat_scores[i][j]
                    inf_text = MathTex(r"-\infty", font_size=24, color=FONDO_CAJA).move_to(bloque_score.get_center())
                    
                    animaciones_infinito.append(bloque_score[0].animate.set_fill(MARRON_OSCURO, opacity=1))
                    animaciones_infinito.append(FadeIn(inf_text))
                    textos_a_reemplazar.append(bloque_score[1])
                    bloque_score[1] = inf_text

        self.play(*[FadeOut(t) for t in textos_a_reemplazar], *animaciones_infinito, run_time=1.5)
        self.next_slide()

        texto_softmax = Text("Resultado: Probabilidad 0 de 'mirar' hacia adelante", 
                             font=FUENTE, font_size=20, weight=BOLD, color=TINTA_NEGRA).move_to(texto_explicativo)
        self.play(Transform(texto_explicativo, texto_softmax))

        animaciones_finales = []
        textos_viejos = []

        for i in range(6):
            for j in range(6):
                bloque = mat_scores[i][j]
                if j > i:
                    cero = Text("0", font=FUENTE, font_size=18, color=MARRON_OSCURO).move_to(bloque.get_center())
                    animaciones_finales.append(bloque[0].animate.set_fill(TINTA_NEGRA, opacity=0.05).set_stroke(MARRON_OSCURO, width=1))
                    animaciones_finales.append(FadeIn(cero))
                    textos_viejos.append(bloque[1])
                else:
                    animaciones_finales.append(bloque[0].animate.set_fill(PAPEL_TAN, opacity=1))

        self.play(*[FadeOut(t) for t in textos_viejos], *animaciones_finales, run_time=1.5)
        
        nota_final = Text("Para predecir 'cobarde', el modelo solo puede usar 'El amor nunca hizo ningún'.", 
                          font=FUENTE, font_size=18, color=TINTA_NEGRA).to_edge(DOWN, buff=0.6)
        
        self.play(FadeIn(nota_final, shift=UP))
        self.next_slide()

        self.limpiar_pantalla()

    def mostrar_acto_arquitectura_neurona(self):
        HIGHLIGHT_COLOR = NARANJA_TERRACOTA 
        
        titulo, linea = self.crear_titulo("La Capa MLP: Una Función Matemática", palabra_clave="Función", color_clave=HIGHLIGHT_COLOR)
        self.play(Write(titulo), Create(linea))

        eq_top = MathTex(r"f : \mathbb{R}^{768} \rightarrow \mathbb{R}^{768}", color=TINTA_NEGRA, font_size=42).next_to(linea, DOWN, buff=0.3)
        self.play(Write(eq_top))

        r = 0.15
        c_linea = GRAY_B
        
        capa_in = VGroup(*[Circle(radius=r, fill_color=PAPEL_TAN, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2) for _ in range(5)]).arrange(DOWN, buff=0.2)
        
        hid_1 = VGroup(*[Circle(radius=r, fill_color=HIGHLIGHT_COLOR, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2) for _ in range(7)]).arrange(DOWN, buff=0.1)
        hid_2 = VGroup(*[Circle(radius=r, fill_color=HIGHLIGHT_COLOR, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2) for _ in range(7)]).arrange(DOWN, buff=0.1)
        hid_3 = VGroup(*[Circle(radius=r, fill_color=HIGHLIGHT_COLOR, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2) for _ in range(7)]).arrange(DOWN, buff=0.1)

        capas_profundas = VGroup(hid_1, hid_2, hid_3).arrange(RIGHT, buff=0.8)
        capa_out = VGroup(*[Circle(radius=r, fill_color=MARRON_OSCURO, fill_opacity=1, stroke_color=TINTA_NEGRA, stroke_width=2) for _ in range(5)]).arrange(DOWN, buff=0.2)

        red = VGroup(capa_in, capas_profundas, capa_out).arrange(RIGHT, buff=1.8).move_to(DOWN * 0.5)

        conexiones_in_h1 = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=c_linea, stroke_opacity=0.3) for n1 in capa_in for n2 in hid_1])
        conexiones_h1_h2 = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=c_linea, stroke_opacity=0.3) for n1 in hid_1 for n2 in hid_2])
        conexiones_h2_h3 = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=c_linea, stroke_opacity=0.3) for n1 in hid_2 for n2 in hid_3])
        conexiones_h3_out = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=c_linea, stroke_opacity=0.3) for n1 in hid_3 for n2 in capa_out])

        brace_in = Brace(capa_in, direction=LEFT, color=MARRON_OSCURO)
        lbl_in = MathTex(r"x \in \mathbb{R}^{768}", color=TINTA_NEGRA, font_size=28).next_to(brace_in, LEFT)
        
        brace_hid = Brace(capas_profundas, direction=UP, color=HIGHLIGHT_COLOR)
        lbl_hid = MathTex(r"3072", color=HIGHLIGHT_COLOR, font_size=32).next_to(brace_hid, UP)
        
        brace_out = Brace(capa_out, direction=RIGHT, color=MARRON_OSCURO)
        lbl_out = MathTex(r"f(x) \in \mathbb{R}^{768}", color=TINTA_NEGRA, font_size=28).next_to(brace_out, RIGHT)

        flecha_1 = MathTex(r"\xrightarrow{\quad \phi_1 \quad}", color=HIGHLIGHT_COLOR).next_to(capas_profundas[0], DOWN, buff=0.5)
        flecha_2 = MathTex(r"\xrightarrow{\quad \phi_2 \quad}", color=HIGHLIGHT_COLOR).next_to(capas_profundas[1], DOWN, buff=0.5)
        flecha_3 = MathTex(r"\xrightarrow{\quad \phi_3 \quad}", color=HIGHLIGHT_COLOR).next_to(capas_profundas[2], DOWN, buff=0.5)
        eq_composicion = MathTex(r"f(x) = \phi_3(\phi_2(\phi_1(x)))", color=TINTA_NEGRA, font_size=32).next_to(VGroup(flecha_1, flecha_3), DOWN, buff=0.3)


        self.play(FadeIn(capa_in), GrowFromCenter(brace_in), Write(lbl_in))
        self.next_slide() 

        self.play(LaggedStartMap(Create, conexiones_in_h1, lag_ratio=0.01), run_time=0.8)
        self.play(FadeIn(hid_1, shift=LEFT*0.2), Write(flecha_1))
        
        self.play(LaggedStartMap(Create, conexiones_h1_h2, lag_ratio=0.01), run_time=0.6)
        self.play(FadeIn(hid_2, shift=LEFT*0.2), Write(flecha_2))
        
        self.play(LaggedStartMap(Create, conexiones_h2_h3, lag_ratio=0.01), run_time=0.6)
        self.play(FadeIn(hid_3, shift=LEFT*0.2), Write(flecha_3))
        
        self.play(GrowFromCenter(brace_hid), Write(lbl_hid), Write(eq_composicion))
        self.next_slide() 

        self.play(LaggedStartMap(Create, conexiones_h3_out, lag_ratio=0.01), run_time=0.8)
        self.play(FadeIn(capa_out, shift=LEFT*0.2), GrowFromCenter(brace_out), Write(lbl_out))
        self.next_slide()

        self.limpiar_pantalla()

    def mostrar_acto_zoom_neurona(self):
        HIGHLIGHT_COLOR = NARANJA_TERRACOTA 
        
        titulo, linea = self.crear_titulo("La Capa MLP: Dentro de una Neurona", palabra_clave="Neurona", color_clave=HIGHLIGHT_COLOR)
        self.play(Write(titulo), Create(linea))

        r_mini = 0.08
        columna_in = VGroup(*[Circle(radius=r_mini, fill_color=PAPEL_TAN, fill_opacity=1, stroke_color=MARRON_OSCURO) for _ in range(3)]).arrange(DOWN, buff=0.1)
        columna_hid = VGroup(*[Circle(radius=r_mini, fill_color=HIGHLIGHT_COLOR, fill_opacity=1, stroke_color=MARRON_OSCURO) for _ in range(5)]).arrange(DOWN, buff=0.1)
        
        mini_red = VGroup(columna_in, columna_hid).arrange(RIGHT, buff=0.6)

        mini_red.to_edge(LEFT, buff=0.8) 
        
        conexiones_mini = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=0.5, color=GRAY_B) for n1 in columna_in for n2 in columna_hid])
        
        self.play(FadeIn(mini_red), FadeIn(conexiones_mini))

        neurona_objetivo = columna_hid[2]
        resaltador = Circle(radius=r_mini * 2, color=MARRON_OSCURO, stroke_width=3).move_to(neurona_objetivo)
        self.play(Create(resaltador))
        self.next_slide()

        posicion_neurona = RIGHT * 2.2 + UP * 0.4
        neurona_gigante = Circle(radius=1.6, fill_color=HIGHLIGHT_COLOR, fill_opacity=0.1, stroke_color=HIGHLIGHT_COLOR, stroke_width=4).move_to(posicion_neurona)
        
        sumatoria = MathTex(r"\Sigma", color=TINTA_NEGRA, font_size=55).move_to(neurona_gigante).shift(LEFT * 0.6)
        separador = Line(neurona_gigante.get_top() + DOWN*0.1, neurona_gigante.get_bottom() + UP*0.1, color=HIGHLIGHT_COLOR)
        
        ejes_act = Axes(x_range=[-2, 2], y_range=[-0.5, 1.5], x_length=1.0, y_length=1.0, axis_config={"color": MARRON_OSCURO, "include_ticks": False}).move_to(neurona_gigante).shift(RIGHT * 0.6)
        curva_act = ejes_act.plot(lambda x: x if x > 0 else 0, color=MARRON_OSCURO) 
        grupo_activacion = VGroup(ejes_act, curva_act)

        lineas_zoom = VGroup(
            DashedLine(resaltador.get_top(), neurona_gigante.get_top(), color=GRAY_B),
            DashedLine(resaltador.get_bottom(), neurona_gigante.get_bottom(), color=GRAY_B)
        )
        
        self.play(Create(lineas_zoom), FadeIn(neurona_gigante))
        self.play(Write(sumatoria), Create(separador), FadeIn(grupo_activacion))
        self.next_slide()

        self.play(FadeOut(lineas_zoom))

        entradas_text = VGroup(
            MathTex(r"x_1", color=TINTA_NEGRA),
            MathTex(r"x_2", color=TINTA_NEGRA),
            MathTex(r"\vdots", color=TINTA_NEGRA), 
            MathTex(r"x_n", color=TINTA_NEGRA)
        ).arrange(DOWN, buff=0.4)
        
        entradas_text.move_to(LEFT * 2.0 + UP * 0.4)

        flechas_in = VGroup(*[Arrow(e.get_right(), neurona_gigante.get_left(), buff=0.15, color=MARRON_OSCURO, max_tip_length_to_length_ratio=0.08) for e in entradas_text if e.tex_string != r"\vdots"])
        
        pesos_text = VGroup(
            MathTex(r"w_1", color=HIGHLIGHT_COLOR).move_to(flechas_in[0].get_center() + UP * 0.35).scale(0.8),
            MathTex(r"w_2", color=HIGHLIGHT_COLOR).move_to(flechas_in[1].get_center() + DOWN * 0.2).scale(0.8),
            MathTex(r"w_n", color=HIGHLIGHT_COLOR).move_to(flechas_in[2].get_center() + DOWN * 0.35).scale(0.8)
        )

        bias_text = MathTex(r"b", color=MARRON_OSCURO).next_to(neurona_gigante, UP, buff=0.4)
        flecha_bias = Arrow(bias_text.get_bottom(), neurona_gigante.get_top(), buff=0.1, color=MARRON_OSCURO)

        flecha_out = Arrow(neurona_gigante.get_right(), neurona_gigante.get_right() + RIGHT * 1.2, color=MARRON_OSCURO)
        salida_text = MathTex(r"\phi(x)", color=TINTA_NEGRA).next_to(flecha_out, RIGHT)

        self.play(LaggedStart(
            *[AnimationGroup(FadeIn(e, shift=RIGHT), GrowArrow(f)) for e, f in zip([entradas_text[0], entradas_text[1], entradas_text[3]], flechas_in)],
            FadeIn(entradas_text[2]), 
            lag_ratio=0.2
        ))
        self.play(FadeIn(pesos_text, shift=UP))
        self.play(FadeIn(bias_text, shift=DOWN), GrowArrow(flecha_bias))
        self.play(GrowArrow(flecha_out), Write(salida_text))
        self.next_slide()

        eq_final = MathTex(r"\phi(x)", r"=", r"\sigma", r"(", r"\sum x_i \cdot w_i", r"+ b", r")", color=TINTA_NEGRA, font_size=46).move_to(DOWN * 1.8)
        
        eq_final[2].set_color(MARRON_OSCURO) 
        eq_final[4].set_color(HIGHLIGHT_COLOR) 
        eq_final[5].set_color(MARRON_OSCURO) 

        nota_w = Text("Suma ponderada\n(Lo que la red aprende)", font=FUENTE, font_size=16, color=HIGHLIGHT_COLOR).next_to(eq_final[4], DOWN, buff=0.6)
        flecha_w = Arrow(nota_w.get_top(), eq_final[4].get_bottom(), buff=0.1, color=HIGHLIGHT_COLOR)
        
        nota_act = Text("Activación no lineal\n(La flexibilidad)", font=FUENTE, font_size=16, color=MARRON_OSCURO).next_to(eq_final[2], DOWN, buff=0.6).shift(LEFT * 0.5)
        flecha_act = Arrow(nota_act.get_top(), eq_final[2].get_bottom(), buff=0.1, color=MARRON_OSCURO)

        self.play(Write(eq_final))
        self.play(FadeIn(nota_w), GrowArrow(flecha_w))
        self.play(FadeIn(nota_act), GrowArrow(flecha_act))
        self.next_slide() 

        self.limpiar_pantalla()

    def mostrar_acto_activacion(self):
        COLOR_NODO_BG = PAPEL_CREMA
        COLOR_BORDE = MARRON_OSCURO
        COLOR_TEXTO = TINTA_NEGRA
        COLOR_RESALTE = NARANJA_TERRACOTA
        COLOR_SECUNDARIO = "#8B2500"

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()
        
        titulo, linea = self.crear_titulo(
            "Función de Activación: GELU", 
            palabra_clave="GELU", 
            color_clave=COLOR_RESALTE
        )
        
        self.play(
            Write(titulo), 
            GrowFromCenter(linea),
            FadeIn(sol_fondo),
            FadeIn(herradura_fondo, shift=UP*0.5),
            FadeIn(llanuras_fondo, shift=UP)
        )

        formula_principal = MathTex(
            r"\text{GELU}(x) = x \cdot \Phi(x)",
            color=COLOR_TEXTO, font_size=42
        )
        formula_aprox = MathTex(
            r"\approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715x^3)\right)\right)",
            color=COLOR_BORDE, font_size=26
        )
        
        grupo_formulas = VGroup(formula_principal, formula_aprox).arrange(DOWN, buff=0.4)
        
        texto_contexto = Text(
            "Apagado de las neuronas", 
            font=FUENTE, font_size=20, color=COLOR_RESALTE, line_spacing=1.2
        )
        
        grupo_izq = VGroup(grupo_formulas, texto_contexto).arrange(DOWN, buff=0.8)
        grupo_izq.to_edge(LEFT, buff=1.0).shift(UP * 1.0)

        self.play(Write(formula_principal))
        self.play(FadeIn(formula_aprox, shift=UP))
        self.play(FadeIn(texto_contexto, shift=RIGHT))

        ejes = Axes(
            x_range=[-3, 3, 1], y_range=[-1, 3, 1],
            x_length=6, y_length=4.5,
            axis_config={"color": COLOR_BORDE, "include_numbers": True, "font_size": 16}
        ).to_edge(RIGHT, buff=0.8).shift(DOWN * 0.5)

        curva_relu = ejes.plot(lambda x: np.maximum(0, x), color=GRAY_C, stroke_width=4)
        lbl_relu = Text("ReLU", font=FUENTE, font_size=26, color=GRAY_C).next_to(ejes.c2p(2, 2), UL, buff=0.2)

        curva_gelu = ejes.plot(
            lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
            color=COLOR_RESALTE, stroke_width=5
        )
        lbl_gelu = Text("GELU", font=FUENTE, font_size=26, color=COLOR_RESALTE).next_to(ejes.c2p(2.5, 2.5), DR, buff=0.1)
        
        punto_minimo = ejes.c2p(-0.75, -0.17)
        lbl_suavizado = Text("Apagado suave", font=FUENTE, font_size=16, color=COLOR_SECUNDARIO).next_to(punto_minimo, DOWN, buff=0.5).shift(RIGHT * 1)
        flecha_suav = Arrow(lbl_suavizado.get_left(), punto_minimo, buff=0.1, color=COLOR_SECUNDARIO, tip_length=0.15)

        self.play(Create(ejes), Write(lbl_relu))
        self.play(Create(curva_relu))
        self.next_slide()

        nodo = Circle(radius=0.7, fill_color=COLOR_NODO_BG, fill_opacity=1, stroke_color=GRAY_C, stroke_width=4)
        lbl_nodo = Text("ReLU", font=FUENTE, font_size=20, color=GRAY_C).move_to(nodo)
        
        flecha_in = Arrow(ORIGIN, RIGHT * 1.5, buff=0.1, color=COLOR_BORDE)
        val_in = MathTex("x = -1", font_size=28, color=COLOR_TEXTO).next_to(flecha_in, UP, buff=0.1)
        grupo_in = VGroup(val_in, flecha_in)
        
        flecha_out = Arrow(ORIGIN, RIGHT * 1.5, buff=0.1, color=GRAY_C)
        val_out = MathTex("0", font_size=32, color=GRAY_C).next_to(flecha_out, UP, buff=0.1)
        grupo_out = VGroup(val_out, flecha_out)

        diagrama_neurona = VGroup(grupo_in, nodo, grupo_out).arrange(RIGHT, buff=0.1)
        diagrama_neurona.to_corner(DL, buff=1.0).shift(UP * 0.5)
        
        lbl_nodo.move_to(nodo)
        
        cruz_muerte = Cross(val_out, stroke_color=RED, stroke_width=5, scale_factor=0.6)

        self.play(FadeIn(nodo), Write(lbl_nodo))
        self.play(GrowArrow(flecha_in), FadeIn(val_in, shift=RIGHT))
        
        val_in_anim = val_in.copy()
        self.play(val_in_anim.animate.move_to(nodo).scale(0.5).set_opacity(0), run_time=0.8)
        
        self.play(GrowArrow(flecha_out), FadeIn(val_out, shift=RIGHT))
        self.play(Create(cruz_muerte))
        self.next_slide()

        lbl_nodo_gelu = Text("GELU", font=FUENTE, font_size=20, color=COLOR_RESALTE).move_to(nodo)
        
        self.play(
            ReplacementTransform(curva_relu, curva_gelu),
            ReplacementTransform(lbl_relu, lbl_gelu),
            nodo.animate.set_stroke(color=COLOR_RESALTE),
            ReplacementTransform(lbl_nodo, lbl_nodo_gelu),
            flecha_out.animate.set_color(COLOR_RESALTE),
            FadeOut(cruz_muerte)
        )
        
        val_out_gelu = MathTex("-0.15", font_size=32, color=COLOR_RESALTE).next_to(flecha_out, UP, buff=0.1)
        chispa = Star(n=5, outer_radius=0.25, inner_radius=0.12, color=COLOR_SECUNDARIO, fill_opacity=1).next_to(val_out_gelu, RIGHT, buff=0.2)

        val_in_anim_2 = val_in.copy()
        self.play(val_in_anim_2.animate.move_to(nodo).scale(0.5).set_opacity(0), run_time=0.8)

        self.play(
            ReplacementTransform(val_out, val_out_gelu),
            Create(chispa)
        )
        
        self.play(FadeIn(lbl_suavizado, shift=UP), GrowArrow(flecha_suav))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_capa_transformer(self):
        COLOR_NODO_BG = PAPEL_TAN
        COLOR_BORDE = MARRON_OSCURO
        COLOR_TEXTO = TINTA_NEGRA
        COLOR_RESALTE = NARANJA_TERRACOTA
        
        escala = 0.65 

        def crear_nodo(texto, ancho=2.5 * escala, alto=0.8 * escala, resaltado=False):
            bg = COLOR_RESALTE if resaltado else COLOR_NODO_BG
            borde = MARRON_OSCURO
            txt_color = WHITE if resaltado else COLOR_TEXTO
            
            caja = RoundedRectangle(
                corner_radius=0.15 * escala, width=ancho, height=alto, 
                fill_color=bg, fill_opacity=1, stroke_color=borde, stroke_width=2.5 * escala
            )
            txt = Text(texto, font=FUENTE, font_size=20 * escala, color=txt_color)
            return VGroup(caja, txt)

        titulo, linea = self.crear_titulo("Arquitectura: Transformer Layer", palabra_clave="Transformer", color_clave=COLOR_RESALTE)
        self.play(Write(titulo), Create(linea))
        
        pos_explicacion = RIGHT * 0.5 + UP * 1.5 

        EJE_X_RESIDUAL = LEFT * 3.5
        EJE_X_BLOQUES = LEFT * 0.5

        nodo_input = crear_nodo("Input").move_to(UP * 2.5 * escala + EJE_X_RESIDUAL)
        
        add_1 = VGroup(
            Circle(radius=0.25 * escala, fill_color=COLOR_NODO_BG, fill_opacity=1, stroke_color=COLOR_BORDE, stroke_width=2 * escala),
            Text("+", font_size=24 * escala, color=COLOR_TEXTO, weight=BOLD)
        ).move_to(UP * 0.2 * escala + EJE_X_RESIDUAL)
        
        add_2 = VGroup(
            Circle(radius=0.25 * escala, fill_color=COLOR_NODO_BG, fill_opacity=1, stroke_color=COLOR_BORDE, stroke_width=2 * escala),
            Text("+", font_size=24 * escala, color=COLOR_TEXTO, weight=BOLD)
        ).move_to(DOWN * 2.2 * escala + EJE_X_RESIDUAL)
        
        nodo_output = crear_nodo("Output").move_to(DOWN * 3.3 * escala + EJE_X_RESIDUAL)

        nodo_ln1 = crear_nodo("Layer Norm").move_to(UP * 1.4 * escala + EJE_X_BLOQUES)
        nodo_attn = crear_nodo("Self-Attention", resaltado=True).move_to(UP * 0.2 * escala + EJE_X_BLOQUES)
        
        nodo_ln2 = crear_nodo("Layer Norm").move_to(DOWN * 1.0 * escala + EJE_X_BLOQUES)
        nodo_mlp = crear_nodo("MLP", resaltado=True).move_to(DOWN * 2.2 * escala + EJE_X_BLOQUES)

        S_WIDTH = 2.5 * escala

        linea_baja_input = Line(nodo_input.get_bottom(), add_1.get_top(), stroke_color=COLOR_BORDE, stroke_width=S_WIDTH)
        punto_split_1 = linea_baja_input.point_from_proportion(0.3)
        flecha_res_1 = Arrow(punto_split_1, add_1.get_top(), buff=0, color=COLOR_BORDE, stroke_width=S_WIDTH, max_tip_length_to_length_ratio=0.15)
        txt_res_1 = Text("Residual Stream", font=FUENTE, font_size=12 * escala, color=GRAY_B).rotate(PI/2).next_to(flecha_res_1, LEFT, buff=0.1)

        esquina_1 = [nodo_ln1.get_center()[0], punto_split_1[1], 0]
        linea_der_1 = Line(punto_split_1, esquina_1, stroke_color=COLOR_BORDE, stroke_width=S_WIDTH)
        flecha_ln1 = Arrow(esquina_1, nodo_ln1.get_top(), buff=0, color=COLOR_BORDE, stroke_width=S_WIDTH)
        flecha_attn = Arrow(nodo_ln1.get_bottom(), nodo_attn.get_top(), buff=0, color=COLOR_BORDE, stroke_width=S_WIDTH)
        flecha_retorno_1 = Arrow(nodo_attn.get_left(), add_1.get_right(), buff=0, color=COLOR_BORDE, stroke_width=S_WIDTH)

        linea_baja_add1 = Line(add_1.get_bottom(), add_2.get_top(), stroke_color=COLOR_BORDE, stroke_width=S_WIDTH)
        punto_split_2 = linea_baja_add1.point_from_proportion(0.3)
        flecha_res_2 = Arrow(punto_split_2, add_2.get_top(), buff=0, color=COLOR_BORDE, stroke_width=S_WIDTH, max_tip_length_to_length_ratio=0.15)
        txt_res_2 = Text("Residual Stream", font=FUENTE, font_size=12 * escala, color=GRAY_B).rotate(PI/2).next_to(flecha_res_2, LEFT, buff=0.1)

        esquina_2 = [nodo_ln2.get_center()[0], punto_split_2[1], 0]
        linea_der_2 = Line(punto_split_2, esquina_2, stroke_color=COLOR_BORDE, stroke_width=S_WIDTH)
        flecha_ln2 = Arrow(esquina_2, nodo_ln2.get_top(), buff=0, color=COLOR_BORDE, stroke_width=S_WIDTH)
        flecha_mlp = Arrow(nodo_ln2.get_bottom(), nodo_mlp.get_top(), buff=0, color=COLOR_BORDE, stroke_width=S_WIDTH)
        flecha_retorno_2 = Arrow(nodo_mlp.get_left(), add_2.get_right(), buff=0, color=COLOR_BORDE, stroke_width=S_WIDTH)

        flecha_out = Arrow(add_2.get_bottom(), nodo_output.get_top(), buff=0, color=COLOR_BORDE, stroke_width=S_WIDTH)

        self.play(FadeIn(nodo_input, shift=DOWN * escala))
        
        self.play(Create(linea_baja_input))
        self.play(Create(linea_der_1), Create(flecha_ln1))
        self.play(FadeIn(nodo_ln1, shift=DOWN * escala))
        self.play(Create(flecha_attn), FadeIn(nodo_attn, shift=DOWN * escala))
        self.play(Create(flecha_retorno_1), FadeIn(add_1, scale=0.5))
        self.play(FadeIn(txt_res_1))
        
        self.play(Create(linea_baja_add1))
        self.play(Create(linea_der_2), Create(flecha_ln2))
        self.play(FadeIn(nodo_ln2, shift=DOWN * escala))
        self.play(Create(flecha_mlp), FadeIn(nodo_mlp, shift=DOWN * escala))
        self.play(Create(flecha_retorno_2), FadeIn(add_2, scale=0.5))
        self.play(FadeIn(txt_res_2))
        
        self.play(Create(flecha_out), FadeIn(nodo_output, shift=UP * escala))
        self.next_slide()
        
        def crear_bloque_texto(titulo, items):
            head = Text(titulo, font=FUENTE, font_size=28, color=COLOR_RESALTE, weight=BOLD)
            bullets = VGroup(*[Text(f"• {item}", font=FUENTE, font_size=22, color=COLOR_TEXTO) for item in items])
            bullets.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(head, DOWN, aligned_edge=LEFT, buff=0.4)
            return VGroup(head, bullets).move_to(pos_explicacion).align_to(pos_explicacion, LEFT)

        grupo_residual = VGroup(linea_baja_input, add_1, linea_baja_add1, add_2, flecha_out)
        caja1 = SurroundingRectangle(grupo_residual, color=COLOR_RESALTE, buff=0.2, stroke_width=3)
        texto1 = crear_bloque_texto("Residual Stream (Autopista):", [
            "Mantiene la información original intacta.",
            "Permite redes muy profundas sin perder gradiente.",
            "Los bloques suman (+) información."
        ])

        self.play(Create(caja1), FadeIn(texto1, shift=LEFT))
        self.next_slide()
        self.play(FadeOut(caja1), FadeOut(texto1))

        caja2 = VGroup(
            SurroundingRectangle(nodo_ln1, color=COLOR_BORDE, buff=0.1, stroke_width=3),
            SurroundingRectangle(nodo_ln2, color=COLOR_BORDE, buff=0.1, stroke_width=3)
        )
        texto2 = crear_bloque_texto("Pre-Layer Normalization:", [
            "Normaliza antes de entrar al bloque.",
            "Esencial para estabilidad matemática.",
            "Estándar en modelos como GPT."
        ])

        self.play(Create(caja2), FadeIn(texto2, shift=LEFT))
        self.next_slide()
        self.play(FadeOut(caja2), FadeOut(texto2))

        caja3 = VGroup(
            SurroundingRectangle(nodo_attn, color=COLOR_RESALTE, buff=0.15, stroke_width=3),
            SurroundingRectangle(nodo_mlp, color=COLOR_RESALTE, buff=0.15, stroke_width=3)
        )
        texto3 = crear_bloque_texto("Bloques de Cómputo:", [
            "Attention: Relaciona palabras entre sí.",
            "MLP: Procesa palabras individualmente."
        ])

        self.play(Create(caja3), FadeIn(texto3, shift=LEFT))
        self.next_slide()
        self.play(FadeOut(caja3), FadeOut(texto3))

        self.limpiar_pantalla()

    def slide_entrenamiento(self):

        titulo, linea = self.crear_titulo(
            "Entrenamiento: Ajustando Perillas", 
            palabra_clave="Perillas", 
            color_clave=NARANJA_TERRACOTA
        )
        self.play(Write(titulo), GrowFromCenter(linea))

        def crear_caja(texto, ancho=2.5, bg_color=FONDO_CAJA, txt_color=TINTA_NEGRA, 
                       opacidad=1.0, borde_color=MARRON_OSCURO, borde_grosor=1, peso=NORMAL):
            """Crea un VGroup estándar con un rectángulo redondeado y texto."""
            caja = RoundedRectangle(
                corner_radius=0.1, width=ancho, height=0.6, 
                fill_color=bg_color, fill_opacity=opacidad, 
                stroke_color=borde_color, stroke_width=borde_grosor
            )
            txt = Text(texto, font=FUENTE, font_size=16, color=txt_color, weight=peso)
            return VGroup(caja, txt)

        def crear_perilla(angulo):
            """Crea la representación visual de un parámetro ajustable."""
            base = Circle(radius=0.25, fill_color=FONDO_CAJA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2)
            indicador = Line(base.get_center(), base.get_center() + UP*0.25, color=NARANJA_TERRACOTA, stroke_width=4)
            indicador.rotate(angulo, about_point=base.get_center())
            return VGroup(base, indicador)


        estado_ui = crear_caja("Iniciando Motor de Entrenamiento...", ancho=7.5).to_edge(UP, buff=1.2)
        self.play(FadeIn(estado_ui, shift=DOWN))

        EJE_Y = DOWN * 0.2

        self.play(
            estado_ui[1].animate.set_text("FASE 1: Forward Pass (Predicción inicial)"), 
            estado_ui[0].animate.set_stroke(NARANJA_TERRACOTA, width=2)
        )

        lbl_in = Text("Contexto", font=FUENTE, font_size=16, color=TINTA_NEGRA, weight=BOLD)
        tokens_in = VGroup(*[crear_caja(word, ancho=1.1) for word in ["En un", "lugar", "de la"]]).arrange(DOWN, buff=0.1)
        grupo_entrada = VGroup(lbl_in, tokens_in).arrange(DOWN, buff=0.3).move_to(LEFT * 4.5 + EJE_Y)

        modelo_bg = RoundedRectangle(corner_radius=0.2, width=3.5, height=2.6, fill_color=PAPEL_CREMA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2)
        lbl_modelo = Text("Transformer\n(Red Neuronal)", font=FUENTE, font_size=18, color=TINTA_NEGRA, weight=BOLD).next_to(modelo_bg.get_top(), DOWN, buff=0.2)
        
        grupo_perillas = VGroup(
            crear_perilla(PI/4), crear_perilla(-PI/3), crear_perilla(PI)
        ).arrange(RIGHT, buff=0.4).move_to(modelo_bg.get_center() + DOWN*0.1)
        
        lbl_pesos = Text("Parámetros (Pesos)", font=FUENTE, font_size=14, color=TINTA_NEGRA).next_to(grupo_perillas, DOWN, buff=0.25)
        grupo_modelo = VGroup(modelo_bg, lbl_modelo, grupo_perillas, lbl_pesos).move_to(EJE_Y)

        lbl_out = Text("Predicción", font=FUENTE, font_size=16, color=TINTA_NEGRA, weight=BOLD)
        prob_incorrecta = crear_caja("playa? (85%)", bg_color=PAPEL_TAN, opacidad=0.8, borde_color=NARANJA_TERRACOTA, borde_grosor=2)
        prob_correcta = crear_caja("Mancha (2%)", bg_color=CAJA_INFERIOR, txt_color=MARRON_OSCURO, opacidad=0.5, borde_color=CAJA_INFERIOR)
        grupo_probs = VGroup(prob_incorrecta, prob_correcta).arrange(DOWN, buff=0.1)
        grupo_salida = VGroup(lbl_out, grupo_probs).arrange(DOWN, buff=0.3).move_to(RIGHT * 4.5 + EJE_Y)

        flujo_in = DashedLine(grupo_entrada.get_right(), modelo_bg.get_left(), color=MARRON_OSCURO, stroke_width=3)
        tubo_out = Line(modelo_bg.get_right(), grupo_salida.get_left(), color=MARRON_OSCURO, stroke_width=4)

        self.play(FadeIn(grupo_entrada, shift=RIGHT))
        self.play(Create(flujo_in))
        self.play(FadeIn(grupo_modelo, scale=0.9))
        self.play(GrowFromCenter(tubo_out))
        self.play(FadeIn(grupo_salida, shift=LEFT))
        self.next_slide()

        self.play(
            estado_ui[1].animate.set_text("FASE 2: Función de Pérdida (Loss)"), 
            estado_ui[0].animate.set_stroke(NARANJA_TERRACOTA, width=2)
        )

        lbl_target = Text("Target Real:", font=FUENTE, font_size=16, color=TINTA_NEGRA, weight=BOLD)
        caja_target = crear_caja("Mancha (100%)")
        grupo_target = VGroup(lbl_target, caja_target).arrange(DOWN, buff=0.1).next_to(grupo_salida, DOWN, buff=0.4)

        nodo_loss = MathTex(r"\mathcal{L}", font_size=40, color=NARANJA_TERRACOTA)
        medidor_bg = RoundedRectangle(width=1.5, height=0.2, corner_radius=0.1, stroke_color=MARRON_OSCURO, fill_color=FONDO_CAJA, fill_opacity=1)
        medidor_fill = RoundedRectangle(width=1.3, height=0.2, corner_radius=0.1, stroke_width=0, fill_color=NARANJA_TERRACOTA, fill_opacity=1).align_to(medidor_bg, LEFT)
        
        caja_error = VGroup(
            Text("Error cuantificado:", font=FUENTE, font_size=14, color=TINTA_NEGRA),
            VGroup(medidor_bg, medidor_fill),
            Text("¡Pérdida muy alta!", font=FUENTE, font_size=14, color=NARANJA_TERRACOTA, weight=BOLD)
        ).arrange(DOWN, buff=0.1)

        panel_loss = VGroup(nodo_loss, caja_error).arrange(RIGHT, buff=0.4).move_to(DOWN * 2.7)

        self.play(FadeIn(grupo_target, shift=UP))
        
        self.play(
            Indicate(prob_incorrecta, color=NARANJA_TERRACOTA, scale_factor=1.1),
            Indicate(caja_target, color=NARANJA_TERRACOTA, scale_factor=1.1)
        )
        
        self.play(FadeIn(panel_loss, shift=UP))
        self.play(Flash(medidor_fill, color=NARANJA_TERRACOTA, line_length=0.2))
        self.next_slide()

        self.play(
            estado_ui[1].animate.set_text("FASE 3: Backpropagation (Ajuste)"), 
            estado_ui[0].animate.set_stroke(NARANJA_TERRACOTA, width=3)
        )

        tubo_back = Arrow(panel_loss.get_top(), modelo_bg.get_bottom(), color=NARANJA_TERRACOTA, stroke_width=4, buff=0.1)
        lbl_grad = MathTex(r"\nabla W", font_size=24, color=NARANJA_TERRACOTA).next_to(tubo_back, RIGHT, buff=0.1)

        self.play(Create(tubo_back), Write(lbl_grad))

        self.play(
            Rotate(grupo_perillas[0][1], angle=-PI/2, about_point=grupo_perillas[0][0].get_center()),
            Rotate(grupo_perillas[1][1], angle=PI/1.5, about_point=grupo_perillas[1][0].get_center()),
            Rotate(grupo_perillas[2][1], angle=-PI/4, about_point=grupo_perillas[2][0].get_center()),
            modelo_bg.animate.set_stroke(NARANJA_TERRACOTA, width=3),
            run_time=2,
            rate_func=there_and_back_with_pause 
        )

        lbl_pesos_nuevos = Text("Parámetros (Ajustados)", font=FUENTE, font_size=14, color=MARRON_OSCURO, weight=BOLD).move_to(lbl_pesos)

        self.play(
            Transform(lbl_pesos, lbl_pesos_nuevos),
            modelo_bg.animate.set_stroke(MARRON_OSCURO, width=2),
            FadeOut(tubo_back, lbl_grad)
        )
        self.next_slide()

        self.play(
            estado_ui[1].animate.set_text("FASE 4: Nuevo Forward Pass (Éxito)"), 
            estado_ui[0].animate.set_stroke(MARRON_OSCURO, width=2)
        )

        flujo_exito = DashedLine(grupo_entrada.get_right(), modelo_bg.get_left(), color=NARANJA_TERRACOTA, stroke_width=4)
        tubo_exito = Line(modelo_bg.get_right(), grupo_salida.get_left(), color=NARANJA_TERRACOTA, stroke_width=5)

        self.play(
            Indicate(modelo_bg, color=PAPEL_TAN),
            Transform(flujo_in, flujo_exito),
            Transform(tubo_out, tubo_exito),
            run_time=1.5
        )

        prob_correcta_nueva = crear_caja("Mancha (98%)", bg_color=PAPEL_TAN, borde_color=MARRON_OSCURO, borde_grosor=2, peso=BOLD).move_to(prob_incorrecta)
        prob_incorrecta_nueva = crear_caja("playa? (1%)", bg_color=CAJA_INFERIOR, txt_color=MARRON_OSCURO, opacidad=0.5, borde_color=CAJA_INFERIOR).move_to(prob_correcta)

        self.play(
            FadeOut(panel_loss, grupo_target),
            Transform(prob_incorrecta, prob_correcta_nueva),
            Transform(prob_correcta, prob_incorrecta_nueva)
        )
        
        self.play(Wiggle(prob_incorrecta, scale_value=1.05))
        self.next_slide()
        
        self.limpiar_pantalla()

    def slide_descenso_gradiente(self):
        titulo, linea = self.crear_titulo(
            "Descenso del Gradiente: Bajando la Colina", 
            palabra_clave="Gradiente:", 
            color_clave=NARANJA_TERRACOTA
        )
        self.play(Write(titulo), GrowFromCenter(linea))

        ejes = Axes(
            x_range=[-3, 5, 1],
            y_range=[0, 10, 2],
            x_length=6,
            y_length=5,
            axis_config={"color": MARRON_OSCURO, "include_numbers": False}
        ).shift(LEFT * 2 + DOWN * 0.2)

        lbl_x = ejes.get_x_axis_label(Text("Parámetro (w)", font=FUENTE, font_size=14, color=MARRON_OSCURO), edge=RIGHT, direction=DOWN)
        lbl_y = ejes.get_y_axis_label(Text("Costo (L)", font=FUENTE, font_size=14, color=MARRON_OSCURO), edge=UP, direction=UP)

        def func_costo(x):
            return np.sin(x) + 0.3 * (x**2) + 2
            
        def gradiente_costo(x):
            return np.cos(x) + 0.6 * x

        curva = ejes.plot(func_costo, color=NARANJA_TERRACOTA, stroke_width=4)
        area = ejes.get_area(curva, opacity=0.1, color=NARANJA_TERRACOTA)
        
        self.play(Create(ejes), Write(lbl_x), Write(lbl_y))
        self.play(Create(curva), FadeIn(area))
        
        def crear_stickman():
            cabeza = Circle(radius=0.15, color=MARRON_OSCURO, fill_color=PAPEL_CREMA, fill_opacity=1)
            cuerpo = Line(cabeza.get_bottom(), cabeza.get_bottom() + DOWN*0.4, color=MARRON_OSCURO, stroke_width=3)
            brazos = VGroup(
                Line(cuerpo.get_center() + UP*0.1, cuerpo.get_center() + UP*0.3 + LEFT*0.2, color=MARRON_OSCURO, stroke_width=3),
                Line(cuerpo.get_center() + UP*0.1, cuerpo.get_center() + UP*0.3 + RIGHT*0.2, color=MARRON_OSCURO, stroke_width=3)
            )
            piernas = VGroup(
                Line(cuerpo.get_bottom(), cuerpo.get_bottom() + DOWN*0.3 + LEFT*0.2, color=MARRON_OSCURO, stroke_width=3),
                Line(cuerpo.get_bottom(), cuerpo.get_bottom() + DOWN*0.3 + RIGHT*0.2, color=MARRON_OSCURO, stroke_width=3)
            )
            # Bandera de meta
            mastil = Line(brazos[0].get_end(), brazos[0].get_end() + UP*0.5, color=TINTA_NEGRA, stroke_width=2)
            bandera = Polygon(
                mastil.get_end(), 
                mastil.get_end() + DOWN*0.15 + LEFT*0.3, 
                mastil.get_end() + DOWN*0.3, 
                fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_width=0
            )
            return VGroup(cabeza, cuerpo, brazos, piernas, mastil, bandera)

        x_minimo = -0.89
        p_minimo = ejes.c2p(x_minimo, func_costo(x_minimo))
        
        stickman = crear_stickman().scale(0.6).next_to(p_minimo, UP, buff=0)
        lbl_meta = Text("Meta", font=FUENTE, font_size=12, color=NARANJA_TERRACOTA, weight=BOLD).next_to(stickman, UP, buff=0.1)
        
        self.play(FadeIn(stickman, shift=UP), Write(lbl_meta))

        lr = 0.8 
        
        def crear_bullet(texto):
            """Crea un texto con una pequeña viñeta naranja."""
            bullet = Circle(radius=0.04, fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_width=0)
            txt = Text(texto, font=FUENTE, font_size=15, color=TINTA_NEGRA)
            return VGroup(bullet, txt).arrange(RIGHT, buff=0.2)

        item_objetivo = crear_bullet("Objetivo: Encontrar el fondo del valle.")
        item_grad = crear_bullet("Gradiente (\u2207L): Indica la pendiente actual.")
        item_lr = crear_bullet(f"Tasa (\u03b7 = {lr}): Controla el tamaño del salto.")
        
        eq_update = MathTex(r"w \leftarrow w - \eta \cdot \nabla L", font_size=32, color=MARRON_OSCURO)
        
        txt_grad = Text("Inclinación: ---", font=FUENTE, font_size=15, color=NARANJA_TERRACOTA, weight=BOLD)
        txt_paso = Text("Posición w: ---", font=FUENTE, font_size=15, color=MARRON_OSCURO)
        
        bloque_textos = VGroup(
            item_objetivo, 
            item_grad, 
            item_lr, 
            eq_update,
            txt_grad, 
            txt_paso
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT).next_to(ejes, RIGHT, buff=0.8).align_to(ejes, UP).shift(DOWN*0.2)

        self.play(FadeIn(bloque_textos, shift=LEFT))
        self.next_slide()
        x_val = 4.0 
        punto = Dot(ejes.c2p(x_val, func_costo(x_val)), color=TINTA_NEGRA, radius=0.12)
        self.play(FadeIn(punto, scale=0.5))

        for i in range(1, 7):
            grad_actual = gradiente_costo(x_val)
            x_next = x_val - lr * grad_actual
            
            p_actual = ejes.c2p(x_val, func_costo(x_val))
            p_next = ejes.c2p(x_next, func_costo(x_next))
            
            si_fuerte = "(¡Inclinado! Salto grande)" if abs(grad_actual) > 0.8 else "(Plano. Salto pequeño)"
            
            nuevo_txt_grad = Text(f"Inclinación: {grad_actual:.2f} {si_fuerte}", font=FUENTE, font_size=15, color=NARANJA_TERRACOTA, weight=BOLD).move_to(txt_grad, aligned_edge=LEFT)
            nuevo_txt_paso = Text(f"Posición w: {x_val:.2f} \u2192 {x_next:.2f}", font=FUENTE, font_size=15, color=MARRON_OSCURO).move_to(txt_paso, aligned_edge=LEFT)
            
            self.play(
                Transform(txt_grad, nuevo_txt_grad),
                Transform(txt_paso, nuevo_txt_paso),
                run_time=0.5
            )
            
            dx = 0.6
            t_start = ejes.c2p(x_val - dx, func_costo(x_val) - grad_actual * dx)
            t_end = ejes.c2p(x_val + dx, func_costo(x_val) + grad_actual * dx)
            tangente = Line(t_start, t_end, color=MARRON_OSCURO, stroke_width=3).set_opacity(0.6)
            
            self.play(Create(tangente), run_time=0.3)
            
            angulo_salto = -TAU/6 if grad_actual > 0 else TAU/6
            flecha_salto = CurvedArrow(p_actual, p_next, angle=angulo_salto, color=NARANJA_TERRACOTA, stroke_width=3)
            
            self.play(Create(flecha_salto), run_time=0.4)
            
            rastro = Dot(p_actual, color=MARRON_OSCURO, radius=0.06).set_opacity(0.4)
            self.add(rastro)
            
            self.play(
                punto.animate.move_to(p_next),
                FadeOut(tangente),
                FadeOut(flecha_salto),
                run_time=0.6
            )
            
            x_val = x_next

        self.play(
            Wiggle(stickman, scale_value=1.15, rotation_angle=0.05 * TAU),
            Flash(punto, color=NARANJA_TERRACOTA, line_length=0.4, flash_radius=0.3)
        )
        self.next_slide()
        
        self.limpiar_pantalla()

    def slide_backpropagation(self):
        titulo, linea = self.crear_titulo(
            "Backpropagation: Regla de la Cadena", 
            palabra_clave="Regla de la Cadena", 
            color_clave=NARANJA_TERRACOTA
        )
        self.play(Write(titulo), GrowFromCenter(linea))

        lbl_intro = Text(
            "Recordemos la Regla de la Cadena:", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        ).move_to(UP * 1.5)

        eq_cadena = MathTex(
            r"\frac{d}{dx} f(g(x)) = \frac{df}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)",
            font_size=40, color=TINTA_NEGRA
        ).next_to(lbl_intro, DOWN, buff=0.5)
        
        self.play(FadeIn(lbl_intro, shift=DOWN*0.5))
        self.play(Write(eq_cadena), run_time=1.5)
        self.next_slide()

        self.play(
            FadeOut(lbl_intro, shift=UP*0.5), 
            FadeOut(eq_cadena, shift=UP*0.5)
        )

        EJE_Y = UP * 1.5

        def crear_nodo(texto, pos, color_borde, color_fondo, es_texto_plano=False):
            circulo = Circle(radius=0.5, fill_color=color_fondo, fill_opacity=1, stroke_color=color_borde, stroke_width=4)
            circulo.move_to(pos)
            if es_texto_plano:
                etiqueta = Text(texto, font_size=16, color=color_borde, weight=BOLD).move_to(circulo)
            else:
                etiqueta = MathTex(texto, font_size=40, color=color_borde).move_to(circulo)
            return VGroup(circulo, etiqueta), circulo, etiqueta

        grp_mul, nodo_mul, simb_mul = crear_nodo(r"\times", LEFT * 3.5 + EJE_Y, MARRON_OSCURO, FONDO_CAJA)
        grp_sum, nodo_sum, simb_sum = crear_nodo(r"+", LEFT * 0 + EJE_Y, MARRON_OSCURO, FONDO_CAJA)
        grp_gelu, nodo_gelu, simb_gelu = crear_nodo("GELU", RIGHT * 3.5 + EJE_Y, MARRON_OSCURO, FONDO_CAJA, es_texto_plano=True)

        txt_x0 = MathTex("x_0", font_size=32, color=TINTA_NEGRA).move_to(nodo_mul.get_left() + LEFT*1.5 + UP*0.6)
        txt_w0 = MathTex("w_0", font_size=32, color=TINTA_NEGRA).move_to(nodo_mul.get_left() + LEFT*1.5 + DOWN*0.6)
        txt_b  = MathTex("b", font_size=32, color=TINTA_NEGRA).move_to(nodo_sum.get_bottom() + DOWN*1.2)
        txt_y  = MathTex(r"\hat{y}", font_size=36, color=NARANJA_TERRACOTA).move_to(nodo_gelu.get_right() + RIGHT*1.5)

        f_x0 = Line(txt_x0.get_right(), nodo_mul.get_left() + UP*0.2, color=MARRON_OSCURO, z_index=-1).add_tip()
        f_w0 = Line(txt_w0.get_right(), nodo_mul.get_left() + DOWN*0.2, color=MARRON_OSCURO, z_index=-1).add_tip()
        f_b  = Line(txt_b.get_top(), nodo_sum.get_bottom(), color=MARRON_OSCURO, z_index=-1).add_tip()
        f_mul_sum = Line(nodo_mul.get_right(), nodo_sum.get_left(), color=MARRON_OSCURO, z_index=-1).add_tip()
        f_sum_gelu = Line(nodo_sum.get_right(), nodo_gelu.get_left(), color=MARRON_OSCURO, z_index=-1).add_tip()
        f_y = Line(nodo_gelu.get_right(), txt_y.get_left(), color=NARANJA_TERRACOTA, z_index=-1).add_tip()

        lbl_mul_out = MathTex(r"\text{mul}", font_size=20, color=MARRON_OSCURO).next_to(f_mul_sum, UP, buff=0.1)
        lbl_sum_out = MathTex(r"\text{sum}", font_size=20, color=MARRON_OSCURO).next_to(f_sum_gelu, UP, buff=0.1)

        grafo_elementos = VGroup(
            grp_mul, grp_sum, grp_gelu, txt_x0, txt_w0, txt_b, txt_y,
            f_x0, f_w0, f_b, f_mul_sum, f_sum_gelu, f_y, lbl_mul_out, lbl_sum_out
        )

        self.play(LaggedStart(
            AnimationGroup(FadeIn(txt_x0), FadeIn(txt_w0)),
            AnimationGroup(Create(f_x0), Create(f_w0)),
            DrawBorderThenFill(grp_mul),
            AnimationGroup(Create(f_mul_sum), FadeIn(lbl_mul_out)),
            AnimationGroup(FadeIn(txt_b), Create(f_b)),
            DrawBorderThenFill(grp_sum),
            AnimationGroup(Create(f_sum_gelu), FadeIn(lbl_sum_out)),
            DrawBorderThenFill(grp_gelu),
            AnimationGroup(Create(f_y), Write(txt_y)),
            lag_ratio=0.3, run_time=3
        ))

        eq_forward = MathTex(r"\hat{y} = \text{GELU}( \text{sum}( \text{mul}(x_0, w_0), b ) )", 
                              font_size=36, color=TINTA_NEGRA).next_to(grafo_elementos, DOWN, buff=0.8)
        self.play(Write(eq_forward))
        self.next_slide()
        
        lbl_backward = Text("Calculando gradientes", 
                            font=FUENTE, font_size=24, color=NARANJA_TERRACOTA, weight=BOLD).move_to(eq_forward)

        self.play(ReplacementTransform(eq_forward, lbl_backward))

        eq_back_w = MathTex(r"\frac{\partial \hat{y}}{\partial w_0} = \frac{\partial \text{GELU}}{\partial \text{sum}} \cdot \frac{\partial \text{sum}}{\partial \text{mul}} \cdot \frac{\partial \text{mul}}{\partial w_0}",
                            font_size=30, color=TINTA_NEGRA).next_to(lbl_backward, DOWN, buff=0.4)
        eq_back_b = MathTex(r"\frac{\partial \hat{y}}{\partial b} = \frac{\partial \text{GELU}}{\partial \text{sum}} \cdot \frac{\partial \text{sum}}{\partial b}",
                            font_size=30, color=TINTA_NEGRA).next_to(eq_back_w, DOWN, buff=0.3)

        l_flash_1 = Line(nodo_gelu.get_left(), nodo_sum.get_right(), color=NARANJA_TERRACOTA, stroke_width=6, z_index=1)
        l_flash_2 = Line(nodo_sum.get_bottom(), txt_b.get_top(), color=NARANJA_TERRACOTA, stroke_width=6, z_index=1)
        l_flash_3 = Line(nodo_sum.get_left(), nodo_mul.get_right(), color=NARANJA_TERRACOTA, stroke_width=6, z_index=1)
        l_flash_4 = Line(nodo_mul.get_left() + DOWN*0.2, txt_w0.get_right(), color=NARANJA_TERRACOTA, stroke_width=6, z_index=1)

        self.play(Write(eq_back_w[0]), Write(eq_back_b[0]), Indicate(nodo_gelu, color=NARANJA_TERRACOTA))
        self.play(ShowPassingFlash(l_flash_1, time_width=0.5), Indicate(nodo_sum), run_time=1)
        self.play(ShowPassingFlash(l_flash_2, time_width=0.5), Indicate(txt_b), Write(eq_back_b[1:]), run_time=1)
        self.play(ShowPassingFlash(l_flash_3, time_width=0.5), Indicate(nodo_mul), Write(eq_back_w[1:3]), run_time=1)
        self.play(ShowPassingFlash(l_flash_4, time_width=0.5), Indicate(txt_w0), Write(eq_back_w[3:]), run_time=1)
        self.next_slide()

        lbl_update = Text(
            "Descenso de Gradiente", 
            font=FUENTE, font_size=24, color=MARRON_OSCURO, weight=BOLD
        ).move_to(lbl_backward.get_center())

        eq_update_w = MathTex(r"w_0 \leftarrow w_0 - \eta \frac{\partial \hat{y}}{\partial w_0}", font_size=38, color=TINTA_NEGRA)
        eq_update_b = MathTex(r"b \leftarrow b - \eta \frac{\partial \hat{y}}{\partial b}", font_size=38, color=TINTA_NEGRA)
        
        eq_update_w.set_color_by_tex(r"\eta", NARANJA_TERRACOTA)
        eq_update_b.set_color_by_tex(r"\eta", NARANJA_TERRACOTA)

        VGroup(eq_update_w, eq_update_b).arrange(DOWN, buff=0.4).next_to(lbl_update, DOWN, buff=0.4)

        self.play(
            ReplacementTransform(lbl_backward, lbl_update),
            ReplacementTransform(eq_back_w, eq_update_w),
            ReplacementTransform(eq_back_b, eq_update_b),
            run_time=1.5
        )

        caja_w0 = SurroundingRectangle(txt_w0, color=NARANJA_TERRACOTA, buff=0.1, corner_radius=0.1)
        caja_b = SurroundingRectangle(txt_b, color=NARANJA_TERRACOTA, buff=0.1, corner_radius=0.1)
        
        lbl_ajuste_w0 = MathTex(r"\boldsymbol{-\eta \Delta w_0}", font_size=20, color=NARANJA_TERRACOTA).next_to(caja_w0, LEFT, buff=0.2)
        lbl_ajuste_b = MathTex(r"\boldsymbol{-\eta \Delta b}", font_size=20, color=NARANJA_TERRACOTA).next_to(caja_b, LEFT, buff=0.2)

        self.play(
            AnimationGroup(
                AnimationGroup(Create(caja_w0), Create(caja_b)),
                AnimationGroup(FadeIn(lbl_ajuste_w0, shift=RIGHT*0.2), FadeIn(lbl_ajuste_b, shift=RIGHT*0.2)),
                lag_ratio=0.5
            ),
            run_time=1.5
        )
        
        self.play(
            AnimationGroup(
                Indicate(lbl_ajuste_w0, color=NARANJA_TERRACOTA, scale_factor=1.2),
                Indicate(lbl_ajuste_b, color=NARANJA_TERRACOTA, scale_factor=1.2),
            ),
            run_time=1.5
        )
        
        self.next_slide()

        self.play(
            FadeOut(grafo_elementos),
            FadeOut(eq_update_w),
            FadeOut(eq_update_b),
            FadeOut(caja_w0),
            FadeOut(caja_b),
            FadeOut(lbl_update),
            FadeOut(lbl_ajuste_w0),
            FadeOut(lbl_ajuste_b),
            run_time=1
        )

        lbl_nn = Text(
            "Propagación y Ajuste en la Red Completa",
            font=FUENTE, font_size=28, color=MARRON_OSCURO, weight=BOLD
        ).move_to(UP * 2)
        
        self.play(FadeIn(lbl_nn, shift=UP*0.2))

        capas = [3, 4, 2]
        nodos = VGroup()
        
        todas_las_conexiones = VGroup()
        conexiones_back_por_capa = []
        
        for i, num_nodos in enumerate(capas):
            nodos_capa = VGroup(*[Circle(radius=0.25, color=MARRON_OSCURO, fill_color=FONDO_CAJA, fill_opacity=1, stroke_width=3) for _ in range(num_nodos)])
            nodos_capa.arrange(DOWN, buff=0.6)
            nodos_capa.move_to(RIGHT * (i * 3.5 - 3.5) + DOWN * 0.5)
            nodos.add(nodos_capa)

        for i in range(len(capas) - 1):
            grupo_back = VGroup()
            for n1 in nodos[i]:
                for n2 in nodos[i+1]:
                    conexion = Line(n1.get_right(), n2.get_left(), stroke_width=2, color=GRAY, z_index=-1).set_opacity(0.3)
                    todas_las_conexiones.add(conexion)
                    
                    conexion_inv = Line(n2.get_left(), n1.get_right(), stroke_width=4, color=NARANJA_TERRACOTA)
                    grupo_back.add(conexion_inv)
            
            conexiones_back_por_capa.append(grupo_back)

        self.play(
            AnimationGroup(
                *[GrowFromCenter(n) for capa in nodos for n in capa],
                lag_ratio=0.1
            ),
            run_time=1.5
        )
        
        self.play(Create(todas_las_conexiones), run_time=1.5)

        txt_loss = MathTex(r"L(y, \hat{y})", font_size=28, color=RED).next_to(nodos[-1], RIGHT, buff=0.5)
        self.play(Write(txt_loss), Indicate(nodos[-1], color=RED))
        self.play(txt_loss.animate.set_color(NARANJA_TERRACOTA))
        self.next_slide()

        for i in reversed(range(len(capas) - 1)):
            etiquetas_ajuste = VGroup()
            for j, linea_inv in enumerate(conexiones_back_por_capa[i]):
                if j % 2 == 0: 
                    lbl_resta = MathTex(r"\boldsymbol{-\eta \Delta w}", font_size=18, color=NARANJA_TERRACOTA)
                    lbl_resta.move_to(linea_inv.get_center() + UP * 0.25)
                    etiquetas_ajuste.add(lbl_resta)

            destellos = [ShowPassingFlash(l.copy(), time_width=0.5) for l in conexiones_back_por_capa[i]]
            
            self.play(
                AnimationGroup(
                    AnimationGroup(*destellos),               
                    FadeIn(etiquetas_ajuste, shift=UP*0.1),   
                    Indicate(nodos[i], color=NARANJA_TERRACOTA), 
                    lag_ratio=0.2
                ),
                run_time=2
            )
            self.play(FadeOut(etiquetas_ajuste, shift=UP*0.1), run_time=0.5)

        self.next_slide()
        self.limpiar_pantalla()

    def slide_adam(self):
        titulo_intro, linea_intro = self.crear_titulo(
            "Optimizador Adam", 
            palabra_clave="Adam", 
            color_clave=NARANJA_TERRACOTA
        )
        self.play(Write(titulo_intro), GrowFromCenter(linea_intro))

        formula_adam = MathTex(
            r"w_{t + 1} \leftarrow w_{t} - ", 
            r"\frac{\eta}{\sqrt{v_{t}} + \epsilon} ", 
            r"\cdot m_{t}", 
            color=TINTA_NEGRA, font_size=60
        ).shift(UP * 2)

        self.play(Write(formula_adam))
        self.next_slide()

        ejes_m = Axes(x_range=[0, 4], y_range=[0, 2], x_length=4, y_length=2).shift(LEFT*3.2 + DOWN*1.5)
        
        func_m = lambda x: 0.5 * np.cos(3*x) - 0.2 * x + 1.5
        terreno_m = ejes_m.plot(func_m, color=MARRON_OSCURO)
        
        lbl_m = Text("m: Momentum", font_size=24, color=NARANJA_TERRACOTA).next_to(ejes_m, UP)

        bola_gd = Dot(color=MARRON_OSCURO).move_to(ejes_m.c2p(0.5, func_m(0.5)))
        bola_adam = Dot(color=NARANJA_TERRACOTA).move_to(ejes_m.c2p(0.5, func_m(0.5)))

        self.play(
            formula_adam[1].animate.set_opacity(0.3),
            formula_adam[2].animate.set_color(NARANJA_TERRACOTA).scale(1.2),
            Create(ejes_m), Create(terreno_m), Write(lbl_m),
            FadeIn(bola_gd), FadeIn(bola_adam)
        )

        tracker_gd = ValueTracker(0.5)
        tracker_adam = ValueTracker(0.5)

        bola_gd.add_updater(lambda d: d.move_to(ejes_m.c2p(tracker_gd.get_value(), func_m(tracker_gd.get_value()))))
        bola_adam.add_updater(lambda d: d.move_to(ejes_m.c2p(tracker_adam.get_value(), func_m(tracker_adam.get_value()))))

        self.play(
            tracker_gd.animate.set_value(1.2), 
            tracker_adam.animate.set_value(3.2), 
            run_time=2.5, rate_func=smooth
        )
        
        bola_gd.clear_updaters()
        bola_adam.clear_updaters()
        self.next_slide()

        ejes_v = Axes(x_range=[0, 4], y_range=[0, 4], x_length=4, y_length=2).shift(RIGHT*3.2 + DOWN*1.5)
        func_v = lambda x: 0.8 * (x - 2.5)**2 + 0.5
        terreno_v = ejes_v.plot(func_v, color=MARRON_OSCURO)
        
        lbl_v = Text("v: Varianza", font_size=24, color=MARRON_OSCURO).next_to(ejes_v, UP)

        self.play(
            formula_adam[2].animate.set_color(TINTA_NEGRA).set_opacity(0.3).scale(1/1.2),
            formula_adam[1].animate.set_opacity(1).set_color(NARANJA_TERRACOTA).scale(1.2),
            Create(ejes_v), Create(terreno_v), Write(lbl_v)
        )

        x_inicio = 0.8
        punto_base = Dot(ejes_v.c2p(x_inicio, func_v(x_inicio)), color=TINTA_NEGRA)
        self.play(FadeIn(punto_base))

        x_sobreimpulso = 3.8
        flecha_mala = Arrow(
            start=ejes_v.c2p(x_inicio, func_v(x_inicio)),
            end=ejes_v.c2p(x_sobreimpulso, func_v(x_sobreimpulso)),
            color=MARRON_OSCURO, buff=0.1
        )
        lbl_mala = Text("Gradiente puro: Paso gigante", font_size=16, color=MARRON_OSCURO).next_to(flecha_mala, UP)
        
        self.play(GrowArrow(flecha_mala), Write(lbl_mala))
        self.wait(0.5)
        
        x_ideal = 2.0
        flecha_buena = Arrow(
            start=ejes_v.c2p(x_inicio, func_v(x_inicio)),
            end=ejes_v.c2p(x_ideal, func_v(x_ideal)),
            color=NARANJA_TERRACOTA, buff=0.1
        )
        lbl_buena = Text("ADAM adapta el paso: Frena en bajadas fuertes", font_size=16, color=NARANJA_TERRACOTA).next_to(flecha_buena, DOWN)

        self.play(
            ReplacementTransform(flecha_mala, flecha_buena),
            ReplacementTransform(lbl_mala, lbl_buena)
        )
        self.wait(0.5)

        tracker_v = ValueTracker(x_inicio)
        punto_v = Dot(color=NARANJA_TERRACOTA)
        punto_v.add_updater(lambda d: d.move_to(ejes_v.c2p(tracker_v.get_value(), func_v(tracker_v.get_value()))))
        
        self.add(punto_v)
        self.remove(punto_base)
        
        self.play(
            tracker_v.animate.set_value(x_ideal), 
            run_time=2, 
            rate_func=smooth
        )
        punto_v.clear_updaters()
        self.next_slide()

        self.play(FadeOut(VGroup(
            ejes_m, terreno_m, lbl_m, bola_gd, bola_adam, 
            ejes_v, terreno_v, lbl_v, flecha_buena, lbl_buena, punto_v, 
            formula_adam, titulo_intro, linea_intro
        )))

        titulo, linea = self.crear_titulo("ADAM vs GD: El Descenso Final", color_clave=NARANJA_TERRACOTA)
        self.play(Write(titulo), GrowFromCenter(linea))

        def func_costo(x):
            return np.cos(3*x)*0.5 + 0.2 * (x**2) + 2

        ejes_gd = Axes(x_range=[-3, 5, 1], y_range=[0, 8, 2], x_length=5.5, y_length=4, 
                       axis_config={"color": MARRON_OSCURO}).shift(LEFT * 3.4 + DOWN * 0.5)
        ejes_adam = Axes(x_range=[-3, 5, 1], y_range=[0, 8, 2], x_length=5.5, y_length=4, 
                         axis_config={"color": NARANJA_TERRACOTA}).shift(RIGHT * 3.4 + DOWN * 0.5)

        lbl_titulo_gd = Text("Gradient Descent Normal", font_size=24, color=MARRON_OSCURO).next_to(ejes_gd, UP)
        lbl_titulo_adam = Text("Optimizador ADAM", font_size=24, color=NARANJA_TERRACOTA).next_to(ejes_adam, UP)

        self.play(
            Create(VGroup(ejes_gd, ejes_gd.plot(func_costo, color=MARRON_OSCURO))),
            Create(VGroup(ejes_adam, ejes_adam.plot(func_costo, color=NARANJA_TERRACOTA))),
            Write(lbl_titulo_gd),
            Write(lbl_titulo_adam)
        )

        pasos_gd, pasos_adam = [4.0], [4.0]
        x_g, x_a = 4.0, 4.0
        m, v = 0, 0
        
        for t in range(1, 60):
            g_a = -np.sin(3*x_a)*1.5 + 0.4*x_a 
            m = 0.9 * m + 0.1 * g_a
            v = 0.999 * v + 0.001 * (g_a**2)
            m_hat = m / (1 - 0.9**t)
            v_hat = v / (1 - 0.999**t)
            
            if abs(x_a) < 0.05 and t > 25:
                x_a = 0.0
            else:
                x_a -= 0.35 * m_hat / (np.sqrt(v_hat) + 1e-8)
            
            pasos_adam.append(x_a)
            g_g = -np.sin(3*x_g)*1.5 + 0.4*x_g
            x_g -= 0.12 * g_g
            pasos_gd.append(x_g)

        trail_gd_points = [ejes_gd.c2p(x, func_costo(x)) for x in pasos_gd]
        trail_adam_points = [ejes_adam.c2p(x, func_costo(x)) for x in pasos_adam]
        
        linea_rastro_gd = VMobject(color=MARRON_OSCURO).set_points_smoothly(trail_gd_points)
        linea_rastro_adam = VMobject(color=NARANJA_TERRACOTA).set_points_smoothly(trail_adam_points)
        
        punto_gd = Dot(trail_gd_points[0], color=MARRON_OSCURO)
        punto_adam = Dot(trail_adam_points[0], color=NARANJA_TERRACOTA)
        
        self.play(FadeIn(punto_gd), FadeIn(punto_adam))

        self.play(
            MoveAlongPath(punto_gd, linea_rastro_gd),
            Create(linea_rastro_gd),
            MoveAlongPath(punto_adam, linea_rastro_adam),
            Create(linea_rastro_adam),
            run_time=4, rate_func=smooth
        )

        lbl_resultado_gd = Text("Mínimo Local", font_size=20, color=MARRON_OSCURO).next_to(punto_gd, DOWN)
        lbl_resultado_adam = Text("Mínimo Global", font_size=20, color=NARANJA_TERRACOTA).next_to(punto_adam, DOWN)

        self.play(
            Write(lbl_resultado_gd),
            Write(lbl_resultado_adam),
            Flash(punto_adam, color=NARANJA_TERRACOTA)
        )

        self.next_slide()
        
        panel_textos = VGroup(
            Text("¿Por qué ganó ADAM?", font_size=26, color=TINTA_NEGRA, weight=BOLD),
            Text("• El Momentum saltó los baches locales.", font_size=20, color=MARRON_OSCURO),
            Text("• La Varianza controló los pasos bruscos.", font_size=20, color=MARRON_OSCURO)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        fondo_panel = Rectangle(
            width=panel_textos.width + 1, height=panel_textos.height + 0.7, 
            color=PAPEL_CREMA, fill_color=FONDO_CAJA, fill_opacity=1
        )
        panel_final = VGroup(fondo_panel, panel_textos).to_edge(DOWN, buff=0.2)

        self.play(FadeIn(panel_final, shift=UP))
        self.next_slide()
        self.limpiar_pantalla()

    def slide_residual_connections(self):
        titulo, linea = self.crear_titulo("Conexiones Residuales: La Autopista", palabra_clave="Conexiones Residuales:", color_clave=NARANJA_TERRACOTA)
        self.play(Write(titulo), GrowFromCenter(linea))

        lbl_forward = Text("1. Forward Pass (Bifurcación)", font=FUENTE, font_size=18, color=TINTA_NEGRA, weight=BOLD)
        
        eq_forward = MathTex(r"y = x + \text{Capa}(x)", font_size=36, color=MARRON_OSCURO)
        desc_forward = Text("La entrada original 'x' se suma a la salida de la capa.", font=FUENTE, font_size=14, color=TINTA_NEGRA).set_opacity(0.8)
        
        grupo_forward = VGroup(lbl_forward, eq_forward, desc_forward).arrange(DOWN, buff=0.25)
        caja_forward = SurroundingRectangle(grupo_forward, color=MARRON_OSCURO, fill_color=FONDO_CAJA, fill_opacity=1, corner_radius=0.2, buff=0.25)
        
        bloque1_completo = VGroup(caja_forward, grupo_forward).next_to(linea, DOWN, buff=0.3)

        self.play(FadeIn(bloque1_completo, shift=DOWN))
        self.next_slide()

        lbl_backward = Text("2. Backward Pass (Acumulación)", font=FUENTE, font_size=18, color=TINTA_NEGRA, weight=BOLD)
        
        eq_back_izq = MathTex(r"\nabla x =", font_size=36, color=TINTA_NEGRA)
        eq_back_skip = MathTex(r"\nabla y", font_size=36, color=NARANJA_TERRACOTA)
        eq_back_plus = MathTex(r"+", font_size=36, color=TINTA_NEGRA)
        eq_back_capa = MathTex(r"\nabla x_{\text{capa}}", font_size=36, color=MARRON_OSCURO)
        
        grupo_eq_back = VGroup(eq_back_izq, eq_back_skip, eq_back_plus, eq_back_capa).arrange(RIGHT, buff=0.2)
        
        txt_skip = Text("Pasa intacto (¡La Autopista!)", font=FUENTE, font_size=14, color=NARANJA_TERRACOTA, weight=BOLD).next_to(eq_back_skip, DOWN, buff=0.6)
        txt_capa = Text("Gradiente transformado", font=FUENTE, font_size=14, color=MARRON_OSCURO).next_to(eq_back_capa, DOWN, buff=1.0) 
        flecha_skip = Arrow(txt_skip.get_top(), eq_back_skip.get_bottom(), buff=0.1, color=NARANJA_TERRACOTA, stroke_width=3, max_tip_length_to_length_ratio=0.15)
        flecha_capa = Arrow(txt_capa.get_top(), eq_back_capa.get_bottom(), buff=0.1, color=MARRON_OSCURO, stroke_width=3, max_tip_length_to_length_ratio=0.15)

        grupo_textos_back = VGroup(lbl_backward, grupo_eq_back, txt_skip, txt_capa, flecha_skip, flecha_capa)
        
        grupo_eq_back.next_to(lbl_backward, DOWN, buff=0.4)
        txt_skip.next_to(eq_back_skip, DOWN, buff=0.7)
        txt_capa.next_to(eq_back_capa, DOWN, buff=1.1)
        flecha_skip.put_start_and_end_on(txt_skip.get_top() + UP*0.1, eq_back_skip.get_bottom() + DOWN*0.1)
        flecha_capa.put_start_and_end_on(txt_capa.get_top() + UP*0.1, eq_back_capa.get_bottom() + DOWN*0.1)

        caja_backward = SurroundingRectangle(grupo_textos_back, color=PAPEL_TAN, fill_color=PAPEL_CREMA, fill_opacity=1, corner_radius=0.2, buff=0.25)
        bloque2_completo = VGroup(caja_backward, grupo_textos_back).next_to(bloque1_completo, DOWN, buff=0.4)

        self.play(Create(caja_backward), FadeIn(lbl_backward, shift=UP))
        self.play(Write(eq_back_izq), Write(eq_back_plus))
        self.next_slide()

        self.play(FadeIn(eq_back_skip, shift=DOWN))
        self.play(GrowArrow(flecha_skip), FadeIn(txt_skip))
        self.next_slide()

        self.play(FadeIn(eq_back_capa, shift=DOWN))
        self.play(GrowArrow(flecha_capa), FadeIn(txt_capa))
        self.next_slide()

        lbl_impacto = Text("Sin esta suma directa, el gradiente desaparecería en redes profundas.", font=FUENTE, font_size=16, color=TINTA_NEGRA, weight=BOLD)
        caja_impacto = SurroundingRectangle(lbl_impacto, color=NARANJA_TERRACOTA, fill_color=CAJA_INFERIOR, fill_opacity=0.5, corner_radius=0.1, buff=0.2)
        bloque3_completo = VGroup(caja_impacto, lbl_impacto).next_to(bloque2_completo, DOWN, buff=0.5)

        self.play(FadeIn(bloque3_completo, shift=UP))
        self.play(Indicate(bloque3_completo, color=NARANJA_TERRACOTA))
        
        self.next_slide()
        self.limpiar_pantalla()
    
    def slide_training_techniques(self):
        titulo, linea = self.crear_titulo("Estabilizadores y Optimización", palabra_clave="Estabilizadores", color_clave=NARANJA_TERRACOTA)
        self.play(Write(titulo), GrowFromCenter(linea))

        lbl_clip = Text("1. Gradient Clipping (Evitar Explosiones)", font=FUENTE, font_size=22, color=TINTA_NEGRA, weight=BOLD)
        
        eq_clip_izq = MathTex(r"g \leftarrow g \cdot", font_size=42, color=TINTA_NEGRA)
        eq_clip_der = MathTex(r"\min\left(1, \frac{\text{max\_norm}}{||g||_2}\right)", font_size=42, color=NARANJA_TERRACOTA)
        grupo_eq_clip = VGroup(eq_clip_izq, eq_clip_der).arrange(RIGHT, buff=0.2)
        
        nota_clip = Text("Si el gradiente es muy grande, reducimos su magnitud preservando su dirección.", font=FUENTE, font_size=14, color=MARRON_OSCURO)
        
        grupo_clip_total = VGroup(lbl_clip, grupo_eq_clip, nota_clip).arrange(DOWN, buff=0.6)
        caja_clip = SurroundingRectangle(grupo_clip_total, color=PAPEL_TAN, fill_color=FONDO_CAJA, fill_opacity=1, corner_radius=0.2, buff=0.4)
        bloque_clip = VGroup(caja_clip, grupo_clip_total).next_to(linea, DOWN, buff=1)

        self.play(FadeIn(caja_clip), FadeIn(lbl_clip, shift=DOWN))
        self.play(Write(eq_clip_izq), Write(eq_clip_der))
        self.play(FadeIn(nota_clip, shift=UP))
        self.next_slide()
        
        self.play(FadeOut(bloque_clip, shift=UP))

        lbl_drop = Text("2. Dropout (Prevenir Memorización)", font=FUENTE, font_size=22, color=TINTA_NEGRA, weight=BOLD)
        
        eq_drop = MathTex(r"y = x \odot M", font_size=42, color=MARRON_OSCURO)
        eq_bernoulli = MathTex(r"M \sim \text{Bernoulli}(1 - p)", font_size=32, color=NARANJA_TERRACOTA)
        grupo_eq_drop = VGroup(eq_drop, eq_bernoulli).arrange(DOWN, buff=0.3)
        
        nota_drop = Text("Apaga aleatoriamente el 10% de las neuronas (p=0.1) para forzar redundancia.", font=FUENTE, font_size=14, color=MARRON_OSCURO)

        grupo_drop_total = VGroup(lbl_drop, grupo_eq_drop, nota_drop).arrange(DOWN, buff=0.6)
        caja_drop = SurroundingRectangle(grupo_drop_total, color=PAPEL_TAN, fill_color=FONDO_CAJA, fill_opacity=1, corner_radius=0.2, buff=0.4)
        bloque_drop = VGroup(caja_drop, grupo_drop_total).next_to(linea, DOWN, buff=1)

        self.play(FadeIn(caja_drop), FadeIn(lbl_drop, shift=DOWN))
        self.play(Write(eq_drop))
        self.play(FadeIn(eq_bernoulli, shift=UP))
        self.play(FadeIn(nota_drop, shift=UP))
        self.next_slide()
        
        self.play(FadeOut(bloque_drop, shift=UP))

        lbl_adamw = Text("3. AdamW (Decoupled Weight Decay)", font=FUENTE, font_size=22, color=TINTA_NEGRA, weight=BOLD)
        
        txt_wd = Text("Paso 1: Decaimiento de Pesos", font=FUENTE, font_size=12, color=NARANJA_TERRACOTA)
        eq_wd = MathTex(r"\theta_{t} = \theta_{t-1}(1 - \eta \lambda)", font_size=34, color=NARANJA_TERRACOTA)
        grupo_wd = VGroup(txt_wd, eq_wd).arrange(DOWN, buff=0.1)

        txt_mv = Text("Paso 2: Promedios Móviles del Gradiente", font=FUENTE, font_size=12, color=MARRON_OSCURO)
        eq_mv = MathTex(r"m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad | \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2", font_size=28, color=MARRON_OSCURO)
        grupo_mv = VGroup(txt_mv, eq_mv).arrange(DOWN, buff=0.1)

        txt_update = Text("Paso 3: Actualización Final", font=FUENTE, font_size=12, color=TINTA_NEGRA)
        eq_update = MathTex(r"\theta_{t} \leftarrow \theta_{t} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}", font_size=38, color=TINTA_NEGRA)
        grupo_update = VGroup(txt_update, eq_update).arrange(DOWN, buff=0.1)

        grupo_eqs_adam = VGroup(grupo_wd, grupo_mv, grupo_update).arrange(DOWN, buff=0.4)
        
        nota_adamw = Text("A diferencia de Adam normal, la regularización NO se mezcla con el momentum.", font=FUENTE, font_size=14, color=NARANJA_TERRACOTA, weight=BOLD)

        grupo_adamw_total = VGroup(lbl_adamw, grupo_eqs_adam, nota_adamw).arrange(DOWN, buff=0.5)
        caja_adamw = SurroundingRectangle(grupo_adamw_total, color=MARRON_OSCURO, fill_color=PAPEL_CREMA, fill_opacity=1, corner_radius=0.2, buff=0.4)
        bloque_adamw = VGroup(caja_adamw, grupo_adamw_total).next_to(linea, DOWN, buff=0.3)

        self.play(FadeIn(caja_adamw), FadeIn(lbl_adamw, shift=DOWN))
        
        self.play(FadeIn(txt_wd), Write(eq_wd))
        self.next_slide()
        
        self.play(FadeIn(txt_mv), Write(eq_mv))
        self.next_slide()
        
        self.play(FadeIn(txt_update), Write(eq_update))
        self.play(FadeIn(nota_adamw, shift=UP))
        
        self.play(Indicate(grupo_wd, color=NARANJA_TERRACOTA))

        self.next_slide()
        self.limpiar_pantalla()

    def slide_training_metrics(self):
        BG_CAJA = PAPEL_CREMA
        BORDE_CAJA = MARRON_OSCURO
        TXT_PRINCIPAL = TINTA_NEGRA
        TXT_SECUNDARIO = MARRON_OSCURO 
        COLOR_CLAVE = NARANJA_TERRACOTA

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Análisis de Convergencia: Loss vs. PPL", 
            palabra_clave="Análisis", 
            color_clave=COLOR_CLAVE
        )
        
        self.play(
            Write(titulo), 
            GrowFromCenter(linea),
            FadeIn(sol_fondo),
            FadeIn(herradura_fondo, shift=UP*0.5),
            FadeIn(llanuras_fondo, shift=UP)
        )

        loss_eq = MathTex(
            r"L = -\frac{1}{N} \sum_{i=1}^{N} \ln P(x_i)", 
            color=TXT_PRINCIPAL
        ).scale(1.2)
        label_loss = Text("Cross-Entropy Loss", font=FUENTE, font_size=18, color=TXT_SECUNDARIO).next_to(loss_eq, UP, buff=0.2)
        loss_group = VGroup(label_loss, loss_eq)

        ppl_eq = MathTex(
            r"PPL(X) = e^L", 
            color=COLOR_CLAVE
        ).scale(1.8)
        label_ppl = Text("Perplexity", font=FUENTE, font_size=20, color=COLOR_CLAVE, weight=BOLD).next_to(ppl_eq, UP, buff=0.2)
        ppl_group = VGroup(label_ppl, ppl_eq)

        math_block = VGroup(loss_group, ppl_group).arrange(DOWN, buff=1.0).move_to(ORIGIN).shift(DOWN * 0.5)

        self.play(
            FadeIn(label_loss), Write(loss_eq),
            run_time=1
        )
        self.play(
            FadeIn(label_ppl), 
            Write(ppl_eq),
            ppl_eq.animate.set_glow(0.5)
        )
        
        explicacion = Tex(
            r"La PPL mide el ``branching factor'':\\¿Entre cuántas palabras duda el modelo?",
            font_size=28, color=TXT_SECUNDARIO
        ).next_to(math_block, DOWN, buff=0.3)

        self.play(FadeIn(explicacion, shift=UP))
        self.next_slide()

        self.play(FadeOut(loss_group, ppl_group, explicacion))

        ax = Axes(
            x_range=[0, 5, 1], 
            y_range=[0, 150, 50],
            x_length=6,        
            y_length=4,        
            axis_config={"include_tip": False, "color": TXT_SECUNDARIO},
            x_axis_config={"label_direction": DOWN},
            y_axis_config={"label_direction": LEFT}
        ).shift(LEFT * 2 + DOWN * 0.5)
        
        labels_ax = ax.get_axis_labels(
            x_label=Text("Loss", font=FUENTE, font_size=16, color=TXT_PRINCIPAL), 
            y_label=Text("PPL", font=FUENTE, font_size=16, color=COLOR_CLAVE)
        )
        curva = ax.plot(lambda x: np.exp(x), x_range=[0, 5], color=COLOR_CLAVE, stroke_width=4)
        
        dot1 = Dot(ax.c2p(4.5, np.exp(4.5)), color=TXT_SECUNDARIO)
        dot2 = Dot(ax.c2p(1.5, np.exp(1.5)), color=COLOR_CLAVE)
        
        lineas_d1 = ax.get_lines_to_point(
            dot1.get_center(), color=TXT_SECUNDARIO
        ).set_opacity(0.4)
        
        lineas_d2 = ax.get_lines_to_point(
            dot2.get_center(), color=COLOR_CLAVE
        ).set_opacity(0.4)

        txt_caos = Text("ENTRENAMIENTO INICIAL\n(PPL > 100)", font=FUENTE, font_size=16, color=TXT_SECUNDARIO).next_to(dot1, UR)
        txt_orden = Text("BUEN ENTRENAMIENTO\n(PPL < 5)", font=FUENTE, font_size=16, color=COLOR_CLAVE).next_to(dot2, DL)

        self.play(Create(ax), Write(labels_ax))
        self.play(Create(curva), run_time=2)
        self.play(Create(lineas_d1), FadeIn(dot1), Write(txt_caos))
        self.play(Create(lineas_d2), FadeIn(dot2), Write(txt_orden))
        self.next_slide()

        self.play(FadeOut(ax, labels_ax, curva, dot1, dot2, lineas_d1, lineas_d2, txt_caos, txt_orden))

        def crear_mensaje_chat(paso, loss, ppl, texto, color_usuario):
            sombra = RoundedRectangle(corner_radius=0.15, height=2.2, width=8)
            sombra.set_fill(MARRON_OSCURO, opacity=0.1).set_stroke(width=0)
            sombra.shift(RIGHT * 0.08 + DOWN * 0.08)
            
            rect = RoundedRectangle(corner_radius=0.15, height=2.2, width=8)
            rect.set_fill(color=BG_CAJA, opacity=1).set_stroke(color=BORDE_CAJA, width=1.5)
            
            icon = Circle(radius=0.25, color=color_usuario, fill_opacity=1)
            label_icon = Text("M", font=FUENTE, font_size=20, color=WHITE, weight=BOLD).move_to(icon)
            user_icon = VGroup(icon, label_icon).next_to(rect, LEFT, buff=0.3).shift(UP * 0.5)
            
            username = Text("MolineteAI", font=FUENTE, font_size=14, color=TXT_SECUNDARIO).next_to(rect, UP, aligned_edge=LEFT).shift(UP*0.1)

            contenido = Paragraph(
                texto, font=FUENTE, font_size=22, color=TXT_PRINCIPAL, 
                line_spacing=1.3, alignment="left"
            ).scale_to_fit_width(rect.width - 0.8).move_to(rect)

            bubble_group = VGroup(sombra, rect, user_icon, username, contenido)
            
            info = Text(f"Paso: {paso}   |   Loss: {loss}   |   PPL: {ppl}", 
                        font="Monospace", font_size=16, color=color_usuario).next_to(rect, DOWN, buff=0.15, aligned_edge=RIGHT)
            
            full_msg = VGroup(bubble_group, info).center().shift(DOWN * 0.5)
            full_msg.contenido_txt = contenido 
            
            return full_msg

        p1 = crear_mensaje_chat("0000", "8.12", "3360.1", "\"Esta q7e llama8 p0r a#í F0rtun4...\"", TXT_SECUNDARIO)
        p2 = crear_mensaje_chat("4000", "4.45", "85.6", "\"Esta que llaman por ahí Fortuna es una mujer...\"", TXT_PRINCIPAL)
        p3 = crear_mensaje_chat("16000", "2.85", "17.2", "\"Esta que llaman por ahí Fortuna es una mujer borracha...\"", COLOR_CLAVE)

        actual = p1
        self.play(FadeIn(actual, scale=0.9))
        self.next_slide()

        for sig in [p2, p3]:
            self.play(FadeTransform(actual, sig), run_time=1.5)
            actual = sig
            self.next_slide()

        self.limpiar_pantalla()

    def slide_model_in_action(self):
        titulo, linea = self.crear_titulo("Molinete AI: Demostración en Vivo", palabra_clave="Vivo", color_clave=NARANJA_TERRACOTA)
        self.play(Write(titulo), GrowFromCenter(linea))

        terminal_window = RoundedRectangle(corner_radius=0.2, width=11, height=6, color=MARRON_OSCURO, fill_color=FONDO_CAJA, fill_opacity=0.95)
        terminal_header = Rectangle(width=11, height=0.5, color=MARRON_OSCURO, fill_color=MARRON_OSCURO, fill_opacity=1).align_to(terminal_window, UP)
        
        dot_1 = Circle(radius=0.08, color="#FF5F56", fill_color="#FF5F56", fill_opacity=1).move_to(terminal_header.get_left() + RIGHT*0.4)
        dot_2 = Circle(radius=0.08, color="#FFBD2E", fill_color="#FFBD2E", fill_opacity=1).next_to(dot_1, RIGHT, buff=0.15)
        dot_3 = Circle(radius=0.08, color="#27C93F", fill_color="#27C93F", fill_opacity=1).next_to(dot_2, RIGHT, buff=0.15)
        

        terminal_title = Text("demostración en vivo", font_size=14, color=FONDO_CAJA, weight=BOLD).move_to(terminal_header) 
        
        terminal_ui = VGroup(terminal_window, terminal_header, dot_1, dot_2, dot_3, terminal_title)
        terminal_ui.next_to(linea, DOWN, buff=0.4)

        self.play(FadeIn(terminal_ui, shift=UP))

        start_pos = terminal_header.get_bottom() + DOWN*0.4 + LEFT*5.2

        comando = Text("$ cargo run --release", font_size=16, font="Monospace", color=TINTA_NEGRA).move_to(start_pos, aligned_edge=LEFT)
        
        barra_vacia = Text("Cargando el modelo: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0%", font_size=16, font="Monospace", color=TINTA_NEGRA).next_to(comando, DOWN, buff=0.3, aligned_edge=LEFT)

        barra_llena = Text("Cargando el modelo: [████████████████████████████████████████] 100%", font_size=16, font="Monospace", color="#A36536").next_to(comando, DOWN, buff=0.3, aligned_edge=LEFT)

        msg_listo = Text("¡Modelo cargado! Ya puedes chatear.", font_size=16, font="Monospace", color=TINTA_NEGRA).next_to(barra_llena, DOWN, buff=0.3, aligned_edge=LEFT)
        
        prompt_user = Text("Tú: En un lugar de la Mancha", font_size=16, font="Monospace", color=NARANJA_TERRACOTA).next_to(msg_listo, DOWN, buff=0.5, aligned_edge=LEFT)
        prompt_model = Text("Molinete: , de cuyo nombre no quiero acordarme, no ha mucho...", font_size=16, font="Monospace", color=TINTA_NEGRA).next_to(prompt_user, DOWN, buff=0.3, aligned_edge=LEFT)

        self.play(Write(comando), run_time=1)
        self.wait(0.3)
        
        self.play(Write(barra_vacia), run_time=0.5)
        self.wait(0.2)
        self.play(Transform(barra_vacia, barra_llena), run_time=2) 
        self.wait(0.3)
        
        self.play(FadeIn(msg_listo, shift=UP*0.1))
        self.wait(0.5)
        
        self.play(AddTextLetterByLetter(prompt_user), run_time=1.5)
        self.wait(0.5)
        
        self.play(AddTextLetterByLetter(prompt_model), run_time=3.5)
        
        self.wait(1)
        self.next_slide()

        transicion_lbl = Text("Cambiando al Demo", font_size=24, color=NARANJA_TERRACOTA, weight=BOLD)
        caja_transicion = SurroundingRectangle(transicion_lbl, color=MARRON_OSCURO, fill_color=FONDO_CAJA, fill_opacity=1, buff=0.4)
        grupo_transicion = VGroup(caja_transicion, transicion_lbl).move_to(terminal_window.get_center())

        contenido_terminal = VGroup(comando, barra_vacia, msg_listo, prompt_user, prompt_model)
        
        self.play(contenido_terminal.animate.set_opacity(0.1), FadeIn(grupo_transicion, scale=0.8))
        
        self.next_slide()
        self.limpiar_pantalla()

    def slide_final(self):
        gracias = Text("¡Muchas Gracias!", font_size=60, weight=BOLD, color=NARANJA_TERRACOTA).move_to(UP*2.5)
        sub = Text("Por tu asistencia y atención", font_size=24, color=MARRON_OSCURO).next_to(gracias, DOWN)
        
        self.play(Write(gracias), FadeIn(sub, shift=UP))

        base_molino = Polygon(
            [-0.9, -1.5, 0], [0.9, -1.5, 0], [0.5, 1.0, 0], [-0.5, 1.0, 0], 
            color=MARRON_OSCURO, fill_opacity=1
        )
        
        puerta = RoundedRectangle(corner_radius=0.2, width=0.6, height=0.8, color=FONDO_CAJA, fill_opacity=1).move_to(base_molino.get_bottom() + UP*0.4)
        ventana = Circle(radius=0.18, color=FONDO_CAJA, fill_opacity=1).move_to(base_molino.get_center() + UP*0.3)
        techo = Polygon([-0.6, 1.0, 0], [0, 1.8, 0], [0.6, 1.0, 0], color=NARANJA_TERRACOTA, fill_opacity=1)
        
        cuerpo_molino = VGroup(base_molino, puerta, ventana, techo)

        def crear_aspa():
            palo = Line(ORIGIN, UP*2.5, color=MARRON_OSCURO, stroke_width=6)
            vela = Polygon(
                [0.1, 0.4, 0], [0.8, 0.4, 0], [0.8, 2.3, 0], [0.1, 2.3, 0], 
                color=NARANJA_TERRACOTA, fill_opacity=0.9, stroke_width=2, stroke_color=MARRON_OSCURO
            )
            lineas = VGroup(*[Line([0.1, y, 0], [0.8, y, 0], color=FONDO_CAJA, stroke_width=2) for y in np.linspace(0.6, 2.1, 6)])
            return VGroup(palo, vela, lineas)

        aspas = VGroup(*[crear_aspa().rotate(i * 90 * DEGREES, about_point=ORIGIN) for i in range(4)])
        
        centro_aspas = techo.get_bottom() + UP*0.3
        aspas.move_to(centro_aspas)
        eje = Dot(centro_aspas, color=FONDO_CAJA, radius=0.15)
        
        molino_completo = VGroup(cuerpo_molino, aspas, eje).scale(0.8).to_edge(LEFT, buff=2).shift(DOWN*1.2)
        
        nuevo_centro_aspas = aspas.get_center()

        ruta_imagen_qr = r"assets\qr_github_molineteai.png" 
        anim_qr = []
        
        qr_real = ImageMobject(ruta_imagen_qr).scale(0.8)
        ancho_marco = qr_real.width + 0.4
        alto_marco = qr_real.height + 0.4
        fondo_qr = RoundedRectangle(
            corner_radius=0.2, width=ancho_marco, height=alto_marco, 
            color=MARRON_OSCURO, stroke_width=4, fill_color=FONDO_CAJA, fill_opacity=1
        )

        grupo_qr = Group(fondo_qr, qr_real).to_edge(RIGHT, buff=2).shift(DOWN*1.2)
        lbl_qr = Text("Escanea para el material", font_size=18, weight=BOLD, color=MARRON_OSCURO).next_to(grupo_qr, UP, buff=0.3)
            
        anim_qr = [DrawBorderThenFill(fondo_qr), FadeIn(qr_real), Write(lbl_qr)]
            
        self.play(
            FadeIn(cuerpo_molino, shift=UP*0.5), 
            GrowFromCenter(aspas),
            FadeIn(eje),
            *anim_qr,
            run_time=2
        )
        
        self.play(Rotate(aspas, angle=2*PI*4, about_point=nuevo_centro_aspas, run_time=10, rate_func=linear))

    def slide_por_que_rust(self):
        titulo, linea = self.crear_titulo("¿Por qué Rust?", palabra_clave="Rust", color_clave=NARANJA_TERRACOTA)
        COLOR_CONTRA = "#B33A3A"

        molino_adorno = crear_molino().scale(0.6).to_corner(DL, buff=0.8)
        sol_adorno = crear_sol_cervantino().scale(0.6).to_corner(UR, buff=0.8)
        tintero_adorno = crear_tintero_y_pluma().scale(0.6).to_corner(DR, buff=0.8)
        escudo_adorno = crear_escudo_y_lanza().scale(0.6).to_corner(UL, buff=0.8)
        adornos = VGroup(molino_adorno, sol_adorno, tintero_adorno, escudo_adorno)

        logo_py = ImageMobject("assets/logo_python.png").set_height(2)
        py_txt1 = Text("Fácil prototipado", font=FUENTE, font_size=24, color=MARRON_OSCURO)
        py_txt2 = Text("✗ Pausas por GC", font=FUENTE, font_size=24, color=COLOR_CONTRA)
        py_txt3 = Text("✗ Lento en inferencia", font=FUENTE, font_size=24, color=COLOR_CONTRA)
        py_grupo = VGroup(py_txt1, py_txt2, py_txt3).arrange(DOWN, buff=0.3)
        bloque_py = Group(logo_py, py_grupo).arrange(DOWN, buff=0.5)

        logo_cpp = ImageMobject("assets/logo_cpp.png").set_height(2)
        cpp_txt1 = Text("Máximo rendimiento", font=FUENTE, font_size=24, color=MARRON_OSCURO)
        cpp_txt2 = Text("✗ Ya muy visto en AI", font=FUENTE, font_size=24, color=COLOR_CONTRA)
        cpp_txt3 = Text("✗ Riesgo en memoria", font=FUENTE, font_size=24, color=COLOR_CONTRA)
        cpp_grupo = VGroup(cpp_txt1, cpp_txt2, cpp_txt3).arrange(DOWN, buff=0.3)
        bloque_cpp = Group(logo_cpp, cpp_grupo).arrange(DOWN, buff=0.5)

        logo_rust = ImageMobject("assets/logo_rust.png").set_height(2.5)
        rust_txt1 = Text("✓ Rendimiento C++", font=FUENTE, font_size=26, color=TINTA_NEGRA)
        rust_txt2 = Text("✓ Memoria segura", font=FUENTE, font_size=26, color=TINTA_NEGRA)
        rust_txt3 = Text("✓ No tan visto en AI", font=FUENTE, font_size=26, color=TINTA_NEGRA)
        rust_grupo = VGroup(rust_txt1, rust_txt2, rust_txt3).arrange(DOWN, buff=0.3)
        bloque_rust = Group(logo_rust, rust_grupo).arrange(DOWN, buff=0.5)

        contenido_principal = Group(bloque_py, bloque_cpp, bloque_rust).arrange(RIGHT, buff=1.0)
        contenido_principal.next_to(linea, DOWN, buff=0.5)

        if contenido_principal.width > 12.0:
            contenido_principal.width = 12.0
            contenido_principal.next_to(linea, DOWN, buff=0.5)

        self.play(Write(titulo), Create(linea), FadeIn(adornos, lag_ratio=0.2))
        self.next_slide()

        self.play(FadeIn(logo_py, shift=DOWN * 0.5))
        self.play(FadeIn(py_txt1, shift=UP * 0.2))
        self.next_slide()
        self.play(FadeIn(py_txt2), FadeIn(py_txt3))
        self.play(Wiggle(py_txt2), Wiggle(py_txt3))
        self.next_slide()
        
        self.play(logo_py.animate.set_opacity(0.2), py_grupo.animate.set_opacity(0.2))

        self.play(FadeIn(logo_cpp, shift=DOWN * 0.5))
        self.play(FadeIn(cpp_txt1, shift=UP * 0.2))
        self.next_slide()
        self.play(FadeIn(cpp_txt2), FadeIn(cpp_txt3))
        self.play(Wiggle(cpp_txt2), Wiggle(cpp_txt3))
        self.next_slide()

        self.play(logo_cpp.animate.set_opacity(0.2), cpp_grupo.animate.set_opacity(0.2))

        logo_rust.generate_target()
        logo_rust.rotate(-math.pi / 2).scale(0.5)
        self.play(
            FadeIn(logo_rust),
            MoveToTarget(logo_rust),
            run_time=1.2
        )
        self.next_slide()

        self.play(
            LaggedStart(
                FadeIn(rust_txt1, shift=RIGHT * 0.3),
                FadeIn(rust_txt2, shift=RIGHT * 0.3),
                FadeIn(rust_txt3, shift=RIGHT * 0.3),
                lag_ratio=0.3
            ),
            run_time=1.5
        )
        self.next_slide()

        caja_rust = RoundedRectangle(
            corner_radius=0.2,
            width=bloque_rust.width + 0.8,
            height=bloque_rust.height + 0.8,
            stroke_color=NARANJA_TERRACOTA,
            stroke_width=4,
            fill_color=FONDO_CAJA,
            fill_opacity=0.3
        ).move_to(bloque_rust)

        self.play(Create(caja_rust))
        self.play(
            rust_txt1.animate.set_color(NARANJA_TERRACOTA),
            rust_txt2.animate.set_color(NARANJA_TERRACOTA),
            rust_txt3.animate.set_color(NARANJA_TERRACOTA),
            Indicate(logo_rust, color=NARANJA_TERRACOTA, scale_factor=1.1)
        )
        self.next_slide()

        self.limpiar_pantalla()

    def diapo_codigo(self, codigo_fuente: str, titulo_archivo: str = "codigo.rs"):
        from pygments.style import Style
        from pygments.token import Keyword, Name, String, Number, Operator, Punctuation, Token, Comment

        class MiEstiloIDE(Style):

            styles = {
                Token: '#2B2B2B',             
                Keyword: 'bold #8B4513',      
                Keyword.Type: 'bold #5C4033', 
                String: '#A0522D',   
                Number: '#B8860B',             
                Name.Function: '#D2691E',      
                Operator: '#2B2B2B',         
                Punctuation: '#2B2B2B',   
                Comment: 'italic #8B8B83',    
            }

        bloque_codigo = Code(
            code_string=codigo_fuente,
            language="rust",
            formatter_style=MiEstiloIDE, 
            background="rectangle"
        ).scale(0.78)
        
        if len(bloque_codigo) > 0:
            bloque_codigo[0].set_opacity(0)
            
        if len(bloque_codigo) > 1:
            bloque_codigo[1].set_color("#8B8B83")

        HEADER = 0.5
        PADDING_X = 1.0
        PADDING_Y = 0.8

        ANCHO = max(bloque_codigo.width + PADDING_X, 5.0)
        ALTO = bloque_codigo.height + PADDING_Y + HEADER

        editor_bg = RoundedRectangle(
            corner_radius=0.2,
            width=ANCHO, height=ALTO,
            color=MARRON_OSCURO,
            fill_color=FONDO_CAJA,
            fill_opacity=0.9,
        )

        editor_header = Rectangle(
            width=ANCHO, height=HEADER,
            color=MARRON_OSCURO,
            fill_color=MARRON_OSCURO,
            fill_opacity=1,
        ).align_to(editor_bg, UP)

        dot_1 = Circle(radius=0.08, color=FONDO_CAJA, fill_opacity=1).move_to(
            editor_header.get_left() + RIGHT * 0.4
        )
        dot_2 = Circle(radius=0.08, color=FONDO_CAJA, fill_opacity=1).next_to(dot_1, RIGHT, buff=0.15)
        dot_3 = Circle(radius=0.08, color=FONDO_CAJA, fill_opacity=1).next_to(dot_2, RIGHT, buff=0.15)

        file_title = Text(
            titulo_archivo, font_size=18, color=FONDO_CAJA, weight=BOLD
        ).move_to(editor_header)

        editor_ui = VGroup(editor_bg, editor_header, dot_1, dot_2, dot_3, file_title)
        editor_ui.move_to(ORIGIN)
        
        area_util = editor_bg.get_center() + DOWN * (HEADER / 2)
        bloque_codigo.move_to(area_util)

        self.play(FadeIn(editor_ui, shift=UP))
        self.next_slide()

        self.play(FadeIn(bloque_codigo, shift=UP * 0.2), run_time=2.5)
        self.next_slide()
    
    def slide_matmul(self):
        titulo, linea = self.crear_titulo(
            "Multiplicación: El 'Dot Product'", 
            palabra_clave="'Dot Product'", 
            color_clave=NARANJA_TERRACOTA
        )

        camino_mancha = FunctionGraph(lambda x: 0.5 * math.sin(x) - 0.5, color=MARRON_OSCURO).set_opacity(0.15)
        camino_punteado = DashedVMobject(camino_mancha, num_dashes=45, dashed_ratio=0.5)
        
        lanza_fondo = Line(LEFT * 7 + DOWN * 2, RIGHT * 7 + UP * 2, color=NARANJA_TERRACOTA, stroke_width=2).set_opacity(0.15)
        
        decoracion_quijote = VGroup(camino_punteado, lanza_fondo).set_z_index(-2)

        molino_adorno = crear_molino().scale(0.6).to_corner(DL, buff=0.8)
        sol_adorno = crear_sol_cervantino().scale(0.6).to_corner(UR, buff=0.8)
        tintero_adorno = crear_tintero_y_pluma().scale(0.6).to_corner(DR, buff=0.8)
        escudo_adorno = crear_escudo_y_lanza().scale(0.6).to_corner(UL, buff=0.8)
        
        sol_adorno.add_updater(lambda m, dt: m.rotate(dt * 0.15))

        self.play(
            Write(titulo), 
            Create(linea),
            FadeIn(decoracion_quijote, run_time=2),
            LaggedStart(
                FadeIn(sol_adorno, scale=0.5, shift=DOWN * 0.3),
                FadeIn(escudo_adorno, scale=0.5, shift=RIGHT * 0.3),
                FadeIn(molino_adorno, scale=0.5, shift=UP * 0.3),
                FadeIn(tintero_adorno, scale=0.5, shift=LEFT * 0.3),
                lag_ratio=0.2
            )
        )

        val_A = ["1","2","3",  "4","5","6",  "7","8","9"]
        val_B = ["2","1","0",  "0","2","1",  "1","0","3"]
        val_C = ["","","",  "","","",  "","",""]

        mat_A = self.crear_matriz_bloques(3, 3, valores=val_A)
        signo_por = Text("×", font=FUENTE, font_size=40, color=TINTA_NEGRA)
        mat_B = self.crear_matriz_bloques(3, 3, valores=val_B)
        signo_igual = Text("=", font=FUENTE, font_size=40, color=TINTA_NEGRA)
        mat_C = self.crear_matriz_bloques(3, 3, valores=val_C)

        grupo_matmul = VGroup(mat_A, signo_por, mat_B, signo_igual, mat_C).arrange(RIGHT, buff=0.4)
        grupo_matmul.shift(UP * 0.5) 
        
        fila_A = mat_A[0] 
        col_B = VGroup(mat_B[0][2], mat_B[1][2], mat_B[2][2]) 
        celda_C = mat_C[0][2] 

        self.play(FadeIn(grupo_matmul))
        self.next_slide()

        self.play(
            *[b[0].animate.set_fill(PAPEL_TAN, opacity=0.8) for b in fila_A],
            *[b[0].animate.set_fill(PAPEL_TAN, opacity=0.8) for b in col_B]
        )
        self.next_slide()

        calculo_texto = Text(
            "(1 × 0) + (2 × 1) + (3 × 3) = 11", 
            font=FUENTE, font_size=32, color=MARRON_OSCURO,
            t2c={"11": NARANJA_TERRACOTA}
        ).next_to(grupo_matmul, DOWN, buff=1.0)
        
        fila_copia = fila_A.copy()
        col_copia = col_B.copy()

        self.play(
            ReplacementTransform(fila_copia, calculo_texto), 
            ReplacementTransform(col_copia, calculo_texto)
        )
        self.next_slide()

        dot_calc = self.crear_bloque("11", color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=0.8)
        dot_calc.move_to(calculo_texto.get_center())

        self.play(ReplacementTransform(calculo_texto, dot_calc))
        
        self.play(
            dot_calc.animate.move_to(celda_C.get_center()),
            celda_C[0].animate.set_fill(PAPEL_TAN, opacity=0.4) 
        )
        self.play(Indicate(dot_calc, color=PAPEL_TAN, scale_factor=1.2))
        self.next_slide()
        
        def_matmul = Text(
            "Combina cada elemento de la fila con su pareja en la columna.\nSe multiplican y se suman para obtener un único número.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        ).to_edge(DOWN, buff=0.6)
        
        self.play(Write(def_matmul))
        self.next_slide()
        
        self.limpiar_pantalla()

    def slide_temperature(self):
        BG_CAJA = PAPEL_CREMA
        BORDE_CAJA = MARRON_OSCURO
        TXT_PRINCIPAL = TINTA_NEGRA
        TXT_SECUNDARIO = MARRON_OSCURO 
        COLOR_CLAVE = NARANJA_TERRACOTA

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "La Temperatura: Controlando la Locura", 
            palabra_clave="Temperatura", 
            color_clave=COLOR_CLAVE
        )
        
        self.play(
            Write(titulo), 
            GrowFromCenter(linea),
            FadeIn(sol_fondo),
            FadeIn(herradura_fondo, shift=UP*0.5),
            FadeIn(llanuras_fondo, shift=UP)
        )

        label_eq = Text("Softmax con Temperatura (T)", font=FUENTE, font_size=18, color=TXT_SECUNDARIO)
        
        eq_temp = MathTex(
            r"P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}", 
            color=TXT_PRINCIPAL
        ).scale(1.3)
        
        eq_temp.set_color_by_tex("T", COLOR_CLAVE)
        
        math_group = VGroup(label_eq, eq_temp).arrange(DOWN, buff=0.3).move_to(ORIGIN).shift(UP * 0.5)

        self.play(FadeIn(label_eq, shift=DOWN*0.2), Write(eq_temp), run_time=1.5)
        self.play(eq_temp.animate.scale(1.1).set_glow(0.3), rate_func=there_and_back, run_time=1)
        self.next_slide()

        explicacion = Tex(
            r"$T \to 0$: Determinista, conservador (Como Sancho)\\$T > 1$: Aleatorio, creativo, ``alucinaciones'' (Como el Quijote)",
            font_size=28, color=TXT_SECUNDARIO, tex_environment="flushleft"
        ).next_to(math_group, DOWN, buff=0.8)

        self.play(FadeIn(explicacion, shift=UP))
        self.next_slide()

        self.play(FadeOut(math_group, explicacion))

        rect_prompt = RoundedRectangle(corner_radius=0.15, height=1.2, width=8)
        rect_prompt.set_fill(color=MARRON_OSCURO, opacity=0.1).set_stroke(color=TXT_SECUNDARIO, width=1.5)
        
        user_label = Text("Usuario", font=FUENTE, font_size=14, color=TXT_SECUNDARIO).next_to(rect_prompt, UP, aligned_edge=LEFT).shift(DOWN*0.1 + RIGHT*0.2)
        
        texto_prompt = Text(
            "Prompt: \"En un lugar de la Mancha...\"", 
            font=FUENTE, font_size=22, color=TXT_PRINCIPAL, weight=BOLD
        ).move_to(rect_prompt)

        grupo_prompt = VGroup(rect_prompt, user_label, texto_prompt).to_edge(UP, buff=1.5)
        
        self.play(FadeIn(grupo_prompt, shift=DOWN))

        def crear_respuesta(temp_val, intento, texto, color_perfil, titulo_perfil):
            sombra = RoundedRectangle(corner_radius=0.15, height=2.2, width=8)
            sombra.set_fill(MARRON_OSCURO, opacity=0.1).set_stroke(width=0)
            sombra.shift(RIGHT * 0.08 + DOWN * 0.08)
            
            rect = RoundedRectangle(corner_radius=0.15, height=2.2, width=8)
            rect.set_fill(color=BG_CAJA, opacity=1).set_stroke(color=BORDE_CAJA, width=1.5)
            
            icon = Circle(radius=0.25, color=color_perfil, fill_opacity=1)
            label_icon = Text(titulo_perfil[5], font=FUENTE, font_size=20, color=WHITE, weight=BOLD).move_to(icon)
            user_icon = VGroup(icon, label_icon).next_to(rect, LEFT, buff=0.3).shift(UP * 0.5)
            
            username = Text(f"{titulo_perfil} (T={temp_val})", font=FUENTE, font_size=14, color=TXT_SECUNDARIO).next_to(rect, UP, aligned_edge=LEFT).shift(UP*0.1)

            contenido = Paragraph(
                texto, font=FUENTE, font_size=22, color=TXT_PRINCIPAL, 
                line_spacing=1.3, alignment="left"
            ).scale_to_fit_width(rect.width - 0.8).move_to(rect)

            bubble_group = VGroup(sombra, rect, user_icon, username, contenido)
            
            info = Text(f"Generación - Intento #{intento}", 
                        font="Monospace", font_size=16, color=color_perfil).next_to(rect, DOWN, buff=0.15, aligned_edge=RIGHT)
            
            return VGroup(bubble_group, info).next_to(grupo_prompt, DOWN, buff=1.0)

        r_sancho_1 = crear_respuesta("0.1", 1, "\"...de cuyo nombre no quiero acordarme.\"", TXT_SECUNDARIO, "Sancho")
        r_sancho_2 = crear_respuesta("0.1", 2, "\"...de cuyo nombre no quiero acordarme.\"", TXT_SECUNDARIO, "Sancho")
        
        r_quijote_1 = crear_respuesta("1.5", 1, "\"...donde los dragones mecánicos beben aceite de oliva.\"", COLOR_CLAVE, "El Quijote")
        r_quijote_2 = crear_respuesta("1.5", 2, "\"...los molinos me hablan en código binario al amanecer.\"", COLOR_CLAVE, "El Quijote")

        actual = r_sancho_1
        self.play(FadeIn(actual, shift=UP))
        self.next_slide()

        self.play(FadeTransform(actual, r_sancho_2), run_time=1)
        actual = r_sancho_2
        self.next_slide()

        self.play(FadeTransform(actual, r_quijote_1), run_time=1.5)
        actual = r_quijote_1
        self.next_slide()

        self.play(FadeTransform(actual, r_quijote_2), run_time=1)
        self.next_slide()

        self.limpiar_pantalla()

    def slide_residual(self):
        COLOR_NODO_BG = PAPEL_CREMA
        COLOR_BORDE = MARRON_OSCURO
        COLOR_TEXTO = TINTA_NEGRA
        COLOR_RESALTE = NARANJA_TERRACOTA
        COLOR_SECUNDARIO = "#8B2500"  
        
        escala = 0.85 

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Conexiones Residuales: El Atajo de Sancho", 
            palabra_clave="Residuales", 
            color_clave=COLOR_RESALTE
        )
        
        self.play(
            Write(titulo), 
            GrowFromCenter(linea),
            FadeIn(sol_fondo),
            FadeIn(herradura_fondo, shift=UP*0.5),
            FadeIn(llanuras_fondo, shift=UP)
        )

        pos_x = LEFT * 4
        pos_f = LEFT * 0.5
        pos_add = RIGHT * 2.5
        pos_y = RIGHT * 4.5

        label_x = MathTex("x", font_size=48 * escala, color=COLOR_TEXTO).move_to(pos_x)
        
        nodo_f = RoundedRectangle(corner_radius=0.2, width=1.5 * escala, height=2.2 * escala, fill_color=COLOR_SECUNDARIO, fill_opacity=1, stroke_color=WHITE, stroke_width=3)
        label_f = MathTex("f", font_size=48 * escala, color=WHITE).move_to(nodo_f)
        grupo_f = VGroup(nodo_f, label_f).move_to(pos_f)
        
        nodo_add = Circle(radius=0.4 * escala, fill_color=COLOR_SECUNDARIO, fill_opacity=1, stroke_color=WHITE, stroke_width=3)
        label_add = MathTex("+", font_size=40 * escala, color=WHITE).move_to(nodo_add)
        grupo_add = VGroup(nodo_add, label_add).move_to(pos_add)
        
        label_y = MathTex("y", font_size=48 * escala, color=COLOR_TEXTO).move_to(pos_y)
        
        eq_final = MathTex("y = x + f(x)", font_size=42 * escala, color=COLOR_TEXTO).next_to(grupo_f, DOWN, buff=1.2)

        arrow_x_f = Arrow(label_x.get_right(), grupo_f.get_left(), buff=0.2, color=COLOR_BORDE, stroke_width=3)
        arrow_f_add = Arrow(grupo_f.get_right(), grupo_add.get_left(), buff=0.2, color=COLOR_BORDE, stroke_width=3)
        arrow_add_y = Arrow(grupo_add.get_right(), label_y.get_left(), buff=0.2, color=COLOR_BORDE, stroke_width=3)

        punto_inicio_skip = pos_x + RIGHT * 0.8
        p1 = punto_inicio_skip
        p2 = p1 + UP * 2.2
        p3 = np.array([grupo_add.get_center()[0], p2[1], 0])
        p4 = grupo_add.get_top() + UP * 0.1
        
        line_up = Line(p1, p2, color=COLOR_BORDE, stroke_width=3)
        line_across = Line(p2, p3, color=COLOR_BORDE, stroke_width=3)
        arrow_down = Arrow(p3, p4, buff=0, color=COLOR_BORDE, stroke_width=3)
        skip_connection = VGroup(line_up, line_across, arrow_down)

        self.play(FadeIn(label_x, shift=RIGHT))
        self.play(GrowArrow(arrow_x_f), FadeIn(grupo_f, shift=RIGHT))
        self.play(GrowArrow(arrow_f_add), FadeIn(grupo_add, scale=0.5))
        self.play(Create(skip_connection))
        self.play(GrowArrow(arrow_add_y), FadeIn(label_y, shift=RIGHT))
        self.play(Write(eq_final))
        self.next_slide()

        pos_texto = DOWN * 3
        
        txt_desc_1 = Text("La señal original viaja por la red...", font=FUENTE, font_size=24 * escala, color=COLOR_TEXTO).move_to(pos_texto)
        txt_desc_2 = Text("El Quijote f(x) transforma el mundo, pero la señal se desvanece.", font=FUENTE, font_size=24 * escala, color=COLOR_SECUNDARIO).move_to(pos_texto)
        txt_desc_3 = Text("Sancho lleva una copia intacta de la realidad por un 'atajo'.", font=FUENTE, font_size=24 * escala, color=COLOR_RESALTE).move_to(pos_texto)
        txt_desc_4 = Text("La realidad de Sancho y la visión del Quijote se suman.", font=FUENTE, font_size=24 * escala, color=COLOR_BORDE).move_to(pos_texto)

        def crear_imagen_pixelada(resolucion="alta"):
            cuadros = []
            filas, cols = (6, 6) if resolucion == "alta" else (3, 3)
            lado = 0.15 if resolucion == "alta" else 0.3
            colores = [COLOR_RESALTE, COLOR_BORDE, COLOR_SECUNDARIO, "#D4A373"]
            
            for i in range(filas):
                for j in range(cols):
                    color = colores[(i*j) % len(colores)]
                    cuadro = Square(side_length=lado, fill_color=color, fill_opacity=1, stroke_width=0.5, stroke_color=WHITE)
                    cuadros.append(cuadro)
                    
            img = VGroup(*cuadros).arrange_in_grid(rows=filas, cols=cols, buff=0)
            if resolucion == "baja":
                img.set_opacity(0.6) 
            return img

        img_alta_x = crear_imagen_pixelada("alta").move_to(pos_x).shift(UP*1.2)
        img_baja_f = crear_imagen_pixelada("baja").move_to(pos_f).shift(UP*1.2)
        img_alta_copia = crear_imagen_pixelada("alta").move_to(pos_x).shift(UP*1.2)
        img_final_y = crear_imagen_pixelada("alta").move_to(pos_y).shift(UP*1.2)

        txt_input = Text("Imagen Real", font=FUENTE, font_size=18, color=COLOR_BORDE).next_to(img_alta_x, UP)
        self.play(FadeIn(img_alta_x, shift=DOWN), FadeIn(txt_input), FadeIn(txt_desc_1))
        self.next_slide()

        self.play(ReplacementTransform(txt_desc_1, txt_desc_2))
        txt_f = Text("Visión Alterada f(x)", font=FUENTE, font_size=18, color=COLOR_SECUNDARIO).next_to(img_baja_f, UP)
        self.play(
            ReplacementTransform(img_alta_x.copy(), img_baja_f),
            FadeIn(txt_f)
        )
        self.next_slide()

        self.play(ReplacementTransform(txt_desc_2, txt_desc_3))
        txt_skip = Text("La Copia de Sancho", font=FUENTE, font_size=18, color=COLOR_RESALTE).next_to(line_across, UP)
        self.play(FadeIn(txt_skip))
        self.play(
            img_alta_copia.animate.move_to(p2).shift(UP*0.5),
            run_time=0.8
        )
        self.play(
            img_alta_copia.animate.move_to(p3).shift(UP*0.5),
            run_time=1.5
        )
        self.play(
            img_alta_copia.animate.move_to(pos_add).shift(UP*1.2),
            run_time=0.8
        )
        self.next_slide()

        self.play(ReplacementTransform(txt_desc_3, txt_desc_4))
        txt_output = Text("Realidad Recuperada", font=FUENTE, font_size=18, color=COLOR_BORDE).next_to(img_final_y, UP)
        
        self.play(
            FadeOut(img_baja_f, shift=RIGHT),
            img_alta_copia.animate.move_to(pos_y).shift(UP*1.2),
            FadeIn(txt_output)
        )
        
        caja_eq = SurroundingRectangle(eq_final, color=COLOR_RESALTE, buff=0.2, stroke_width=2)
        self.play(Create(caja_eq))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_dropout(self):
        BG_CAJA = PAPEL_CREMA
        TXT_PRINCIPAL = TINTA_NEGRA
        TXT_SECUNDARIO = MARRON_OSCURO 
        COLOR_CLAVE = NARANJA_TERRACOTA
        COLOR_INPUT = BLUE_D
        COLOR_OUTPUT = GREEN_D

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Dropout: El Arte de Olvidar", 
            palabra_clave="Dropout", 
            color_clave=COLOR_CLAVE
        )
        
        self.play(
            Write(titulo), 
            GrowFromCenter(linea),
            FadeIn(sol_fondo),
            FadeIn(herradura_fondo, shift=UP*0.5),
            FadeIn(llanuras_fondo, shift=UP)
        )

        label_proposito = Text(
            "Es una técnica de regularización.", 
            font=FUENTE, font_size=24, color=TXT_PRINCIPAL, weight=BOLD
        )

        label_eq = Text(
            "Se basa en una Distribución de Bernoulli:", 
            font=FUENTE, font_size=20, color=TXT_SECUNDARIO
        )
        
        eq_dropout = MathTex(
            r"r_i \sim \text{Bernoulli}(p)", 
            r"\quad \Rightarrow \quad",
            r"y_i = \frac{r_i \cdot x_i}{p}",
            color=TXT_PRINCIPAL
        ).scale(1.2)
        
        eq_dropout.set_color_by_tex("p", COLOR_CLAVE)
        eq_dropout.set_color_by_tex("r_i", MARRON_OSCURO)

        explicacion = Tex(
            r"Al escalar por $p$, mantenemos la esperanza matemática: $\mathbb{E}[y_i] = x_i$",
            font_size=24, color=TXT_SECUNDARIO
        )
        
        grupo_intro = VGroup(label_proposito, label_eq, eq_dropout, explicacion).arrange(DOWN, buff=0.4).move_to(ORIGIN)

        self.play(FadeIn(grupo_intro, shift=DOWN*0.2), run_time=1.5)
        self.next_slide()

        self.play(FadeOut(grupo_intro))

        tamaños_capas = [4, 5, 5, 5, 4]
        colores_capas = [COLOR_INPUT, COLOR_CLAVE, COLOR_CLAVE, COLOR_CLAVE, COLOR_OUTPUT]
        nombres_capas = ["Input", "Dense 1", "Dense 2", "Dense 3", "Output"]
        
        nodos = VGroup()
        etiquetas = VGroup()
        
        for size, color, nombre in zip(tamaños_capas, colores_capas, nombres_capas):
            capa = VGroup(*[Dot(radius=0.15, color=color) for _ in range(size)]).arrange(DOWN, buff=0.4)
            nodos.add(capa)
            
            etiqueta = Text(nombre, font=FUENTE, font_size=16, color=TXT_SECUNDARIO, weight=BOLD)
            etiqueta.next_to(capa, UP, buff=0.3)
            etiquetas.add(etiqueta)
            
        nodos.arrange(RIGHT, buff=1.8).shift(DOWN * 0.2)
        
        for i, etiqueta in enumerate(etiquetas):
            etiqueta.next_to(nodos[i], UP, buff=0.3)

        conexiones = VGroup()
        for i in range(len(tamaños_capas) - 1):
            capa_act = nodos[i]
            capa_sig = nodos[i+1]
            grupo_conexiones = VGroup()
            for n1 in capa_act:
                for n2 in capa_sig:
                    grupo_conexiones.add(Line(n1.get_center(), n2.get_center(), stroke_width=1.5, color=TXT_SECUNDARIO, stroke_opacity=0.3))
            conexiones.add(grupo_conexiones)

        red_grupo = VGroup(conexiones, nodos, etiquetas)

        self.play(FadeIn(red_grupo))
        self.next_slide()

        def aplicar_dropout_capa(indice_capa, indices_apagar, texto_explicativo):
            animaciones = []
            capa = nodos[indice_capa]
            
            for idx in indices_apagar:
                animaciones.append(capa[idx].animate.set_color(GRAY).set_opacity(0.2))
                
                if indice_capa > 0:
                    for n1_idx in range(tamaños_capas[indice_capa - 1]):
                        line_idx = n1_idx * tamaños_capas[indice_capa] + idx
                        animaciones.append(conexiones[indice_capa - 1][line_idx].animate.set_stroke(opacity=0.02))
                
                if indice_capa < len(tamaños_capas) - 1:
                    for n2_idx in range(tamaños_capas[indice_capa + 1]):
                        line_idx = idx * tamaños_capas[indice_capa + 1] + n2_idx
                        animaciones.append(conexiones[indice_capa][line_idx].animate.set_stroke(opacity=0.02))
            
            texto = Text(texto_explicativo, font=FUENTE, font_size=20, color=TXT_PRINCIPAL).to_edge(DOWN, buff=0.5)
            
            self.play(*animaciones, FadeIn(texto, shift=UP), run_time=1.5)
            self.next_slide()
            self.play(FadeOut(texto))

        aplicar_dropout_capa(1, [1, 4], "Dense 1: Desactivamos el 40% de las neuronas al azar.")
        aplicar_dropout_capa(2, [0, 2, 3], "Dense 2: Otras neuronas deben aprender a compensar la pérdida.")
        aplicar_dropout_capa(3, [1, 4], "Dense 3: Ninguna neurona se vuelve indispensable.")

        texto_metafora = Text(
            "\"No confíes la batalla a un solo caballero,\nhaz que toda la orden luche unida.\"",
            font=FUENTE, font_size=24, color=TXT_SECUNDARIO, slant=ITALIC
        ).to_edge(DOWN, buff=0.5)

        self.play(Write(texto_metafora))
        self.next_slide()

        self.limpiar_pantalla()