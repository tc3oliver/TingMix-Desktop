//! EXPERIMENTAL MLX backend for TingMix-Large: audio tower + GRU-CTC reader, batch 1, on the MLX default device (GPU).
//!
//! Input is the 128-bin Gemma4 log-mel (computed elsewhere); output is reader logits `[frames, vocab]` ready for CTC
//! decoding. [`MlxLargeBackend`] adds the Rust mel front end, 29.5 s chunking and CTC decoding behind `AsrBackend`.

mod backend;

pub use backend::{Decode, MlxLargeBackend};

#[cfg(mlx_native)]
mod ffi {
    use std::os::raw::{c_char, c_int};

    #[repr(C)]
    pub struct TmModel {
        _p: [u8; 0],
    }
    extern "C" {
        pub fn tm_load(
            path: *const c_char,
            dtype: c_int,
            err: *mut c_char,
            err_len: usize,
        ) -> *mut TmModel;
        pub fn tm_free(m: *mut TmModel);
        pub fn tm_vocab(m: *const TmModel) -> c_int;
        pub fn tm_run(
            m: *mut TmModel,
            mel: *const f32,
            n_mel: c_int,
            n_valid: c_int,
            feats: *mut f32,
            logits: *mut f32,
            err: *mut c_char,
            err_len: usize,
        ) -> c_int;
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MlxError {
    #[error("built without MLX (set MLX_PREFIX on aarch64 macOS)")]
    NotBuilt,
    #[error("mlx: {0}")]
    Mlx(String),
    #[error("invalid input: {0}")]
    Input(&'static str),
}

/// Weight precision applied at load time to the fp32 artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32 = 0,
    F16 = 1,
    Bf16 = 2,
}

pub const FEATURE_DIM: usize = 1536;
pub const MEL_BINS: usize = 128;

/// Encoder frames produced for `n_valid_mel` real mel frames (two stride-2 convolutions).
pub fn out_frames(n_valid_mel: usize) -> usize {
    (n_valid_mel.div_ceil(2)).div_ceil(2)
}

#[cfg(mlx_native)]
pub struct MlxLarge {
    ptr: *mut ffi::TmModel,
    vocab: usize,
}

// SAFETY: MLX arrays are created and consumed inside each `run` call on the calling thread; the handle only holds
// immutable weights after `tm_load`, and `run` takes `&mut self`, so it is never entered concurrently.
#[cfg(mlx_native)]
unsafe impl Send for MlxLarge {}

#[cfg(mlx_native)]
fn msg(buf: &[u8]) -> String {
    let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

#[cfg(mlx_native)]
impl MlxLarge {
    /// Load the single-file fp32 safetensors artifact written by `experiments/macos/mlx/convert.py`.
    pub fn load(path: &std::path::Path, dtype: Dtype) -> Result<Self, MlxError> {
        let c = std::ffi::CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| MlxError::Input("path contains NUL"))?;
        let mut err = [0u8; 512];
        // SAFETY: `c` and `err` outlive the call; tm_load writes at most err.len() bytes.
        let ptr =
            unsafe { ffi::tm_load(c.as_ptr(), dtype as i32, err.as_mut_ptr().cast(), err.len()) };
        if ptr.is_null() {
            return Err(MlxError::Mlx(msg(&err)));
        }
        let vocab = unsafe { ffi::tm_vocab(ptr) } as usize;
        Ok(Self { ptr, vocab })
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// `mel`: row-major `[n_mel, 128]`, the first `n_valid` rows real. Returns logits `[out_frames(n_valid), vocab]`.
    pub fn logits(&mut self, mel: &[f32], n_valid: usize) -> Result<Vec<f32>, MlxError> {
        if mel.is_empty() || mel.len() % MEL_BINS != 0 {
            return Err(MlxError::Input("mel length is not a multiple of 128"));
        }
        let n_mel = mel.len() / MEL_BINS;
        if n_valid == 0 || n_valid > n_mel {
            return Err(MlxError::Input("n_valid out of range"));
        }
        let mut out = vec![0f32; out_frames(n_valid) * self.vocab];
        let mut err = [0u8; 512];
        // SAFETY: buffers are sized per the C header contract; feats is optional (null).
        let rc = unsafe {
            ffi::tm_run(
                self.ptr,
                mel.as_ptr(),
                n_mel as i32,
                n_valid as i32,
                std::ptr::null_mut(),
                out.as_mut_ptr(),
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        if rc != 0 {
            return Err(MlxError::Mlx(msg(&err)));
        }
        Ok(out)
    }
}

#[cfg(mlx_native)]
impl Drop for MlxLarge {
    fn drop(&mut self) {
        // SAFETY: ptr came from tm_load and is freed exactly once.
        unsafe { ffi::tm_free(self.ptr) }
    }
}

#[cfg(not(mlx_native))]
pub struct MlxLarge;

#[cfg(not(mlx_native))]
impl MlxLarge {
    pub fn load(_: &std::path::Path, _: Dtype) -> Result<Self, MlxError> {
        Err(MlxError::NotBuilt)
    }

    pub fn vocab(&self) -> usize {
        0
    }

    pub fn logits(&mut self, _: &[f32], _: usize) -> Result<Vec<f32>, MlxError> {
        Err(MlxError::NotBuilt)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn frame_count_matches_conv_arithmetic() {
        // (T+1)//2 twice, as in the Python/C++ graph: 170 mel frames -> 43 encoder frames
        assert_eq!(super::out_frames(170), 43);
        assert_eq!(super::out_frames(2999), 750);
    }
}
