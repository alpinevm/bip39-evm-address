use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Rerun build script if the CUDA source changes.
    println!("cargo:rerun-if-changed=cuda/main.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let fatbin_path = out_dir.join("kernel.fatbin");

    // Compile the CUDA kernel
    let status = Command::new("nvcc")
        .args(&[
            "-fatbin",
            "-o",
            fatbin_path.to_str().unwrap(),
            "cuda/main.cu",
        ])
        .status()
        .expect("failed to execute nvcc");

    if !status.success() {
        panic!("nvcc failed to compile the CUDA file");
    }
}
