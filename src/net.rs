//! GenoNet inference: log-mel -> 16-param genome. Weights (BatchNorm folded)
//! are embedded in the binary — no model file to ship. See export_net.py.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};

use crate::genome::{Genome, N_PARAMS};

static WEIGHTS_BIN: &[u8] = include_bytes!("../weights/genonet.bin");

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct Net {
    t: HashMap<String, Tensor>,
}

fn read_u32(b: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*o..*o + 4].try_into().unwrap());
    *o += 4;
    v
}

impl Net {
    pub fn load() -> Result<Self> {
        let b = WEIGHTS_BIN;
        if &b[..4] != b"SYNF" {
            bail!("bad weights magic");
        }
        let mut o = 4;
        let _ver = read_u32(b, &mut o);
        let n = read_u32(b, &mut o) as usize;
        let mut t = HashMap::new();
        for _ in 0..n {
            let name_len = u16::from_le_bytes(b[o..o + 2].try_into().unwrap()) as usize;
            o += 2;
            let name = std::str::from_utf8(&b[o..o + name_len])?.to_string();
            o += name_len;
            let ndim = b[o] as usize;
            o += 1;
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                shape.push(read_u32(b, &mut o) as usize);
            }
            let count: usize = shape.iter().product();
            let mut data = Vec::with_capacity(count);
            for i in 0..count {
                data.push(f32::from_le_bytes(b[o + 4 * i..o + 4 * i + 4].try_into().unwrap()));
            }
            o += 4 * count;
            t.insert(name, Tensor { shape, data });
        }
        Ok(Net { t })
    }

    fn get(&self, name: &str) -> Result<&Tensor> {
        self.t.get(name).with_context(|| format!("missing tensor {name}"))
    }

    pub fn mel_fb(&self) -> Result<&Tensor> {
        self.get("melfb")
    }

    /// mel: [n_mels * frames] row-major (mel, frame) -> genome in [0,1].
    pub fn forward(&self, mel: &[f32], h: usize, w: usize) -> Result<Genome> {
        let mut x = mel.to_vec();
        let (mut ch, mut hh, mut ww) = (1usize, h, w);
        for blk in 0..4 {
            let (w1, b1) = (self.get(&format!("b{blk}c1w"))?, self.get(&format!("b{blk}c1b"))?);
            x = conv3x3(&x, ch, hh, ww, w1, b1);
            ch = w1.shape[0];
            gelu(&mut x);
            let (w2, b2) = (self.get(&format!("b{blk}c2w"))?, self.get(&format!("b{blk}c2b"))?);
            x = conv3x3(&x, ch, hh, ww, w2, b2);
            ch = w2.shape[0];
            gelu(&mut x);
            let (nx, nh, nw) = maxpool2(&x, ch, hh, ww);
            x = nx;
            hh = nh;
            ww = nw;
        }
        // global average pool -> (ch,)
        let mut feat = vec![0.0f32; ch];
        let hw = hh * ww;
        for c in 0..ch {
            feat[c] = x[c * hw..(c + 1) * hw].iter().sum::<f32>() / hw as f32;
        }
        let mut y = linear(&feat, self.get("fc1w")?, self.get("fc1b")?);
        gelu(&mut y);
        let y = linear(&y, self.get("fc2w")?, self.get("fc2b")?);
        let mut g = [0.0f32; N_PARAMS];
        for (i, v) in y.iter().enumerate() {
            g[i] = 1.0 / (1.0 + (-v).exp());
        }
        Ok(g)
    }
}

/// Exact GELU (erf form), matching nn.GELU()'s default.
fn gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let f = *v as f64;
        *v = (0.5 * f * (1.0 + libm::erf(f / std::f64::consts::SQRT_2))) as f32;
    }
}

/// 3x3 conv, stride 1, pad 1 (same size). Input/output CHW.
fn conv3x3(x: &[f32], ci: usize, h: usize, w: usize, wt: &Tensor, b: &Tensor) -> Vec<f32> {
    let co = wt.shape[0];
    debug_assert_eq!(wt.shape[1], ci);
    let mut out = vec![0.0f32; co * h * w];
    for oc in 0..co {
        let bias = b.data[oc];
        for oy in 0..h {
            for ox in 0..w {
                let mut acc = bias;
                for ic in 0..ci {
                    let xoff = ic * h * w;
                    let woff = (oc * ci + ic) * 9;
                    for ky in 0..3usize {
                        let iy = oy as isize + ky as isize - 1;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        let row = xoff + iy as usize * w;
                        for kx in 0..3usize {
                            let ix = ox as isize + kx as isize - 1;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            acc += x[row + ix as usize] * wt.data[woff + ky * 3 + kx];
                        }
                    }
                }
                out[oc * h * w + oy * w + ox] = acc;
            }
        }
    }
    out
}

/// 2x2 max pool, stride 2, floor (drops odd edges) — nn.MaxPool2d(2).
fn maxpool2(x: &[f32], ch: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let (oh, ow) = (h / 2, w / 2);
    let mut out = vec![0.0f32; ch * oh * ow];
    for c in 0..ch {
        for oy in 0..oh {
            for ox in 0..ow {
                let (iy, ix) = (oy * 2, ox * 2);
                let base = c * h * w;
                let m = x[base + iy * w + ix]
                    .max(x[base + iy * w + ix + 1])
                    .max(x[base + (iy + 1) * w + ix])
                    .max(x[base + (iy + 1) * w + ix + 1]);
                out[c * oh * ow + oy * ow + ox] = m;
            }
        }
    }
    (out, oh, ow)
}

fn linear(x: &[f32], wt: &Tensor, b: &Tensor) -> Vec<f32> {
    let (out_d, in_d) = (wt.shape[0], wt.shape[1]);
    debug_assert_eq!(x.len(), in_d);
    (0..out_d)
        .map(|o| {
            let row = &wt.data[o * in_d..(o + 1) * in_d];
            b.data[o] + row.iter().zip(x).map(|(w, v)| w * v).sum::<f32>()
        })
        .collect()
}
