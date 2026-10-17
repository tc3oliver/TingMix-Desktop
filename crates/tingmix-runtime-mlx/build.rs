//! Builds `cpp/tingmix_mlx.cpp` against a prebuilt MLX (headers + libmlx.dylib + mlx.metallib).
//! Set `MLX_PREFIX` to a directory holding `include/mlx/` and `lib/libmlx.dylib` (the PyPI `mlx` wheel layout works).
//! Without it (or off Apple silicon) the crate compiles to an empty stub and the `mlx_native` cfg is not set.
use std::env;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(mlx_native)");
    println!("cargo:rerun-if-env-changed=MLX_PREFIX");
    println!("cargo:rerun-if-changed=cpp");
    let apple_arm = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        && env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64");
    let Some(prefix) = env::var_os("MLX_PREFIX").filter(|_| apple_arm) else {
        println!("cargo:warning=tingmix-runtime-mlx: MLX_PREFIX unset or not aarch64 macOS; building stub without MLX");
        return;
    };
    let prefix = std::path::PathBuf::from(prefix);
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .file("cpp/tingmix_mlx.cpp")
        .flag(format!("-isystem{}", prefix.join("include").display()))
        .opt_level(2)
        .compile("tingmix_mlx");
    let lib = prefix.join("lib");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=mlx");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    println!("cargo:rustc-cfg=mlx_native");
}
