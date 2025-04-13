# CUDA Rust Template

A minimal template for calling CUDA kernels from Rust hosts using the `cust` crate.

## Prerequisites

- Rust toolchain (install via [rustup](https://rustup.rs/))
- CUDA toolkit (with `nvcc` in your PATH)
- A CUDA-capable GPU

## Project Structure

```
.
├── Cargo.toml         # Project dependencies
├── build.rs           # Build script to compile CUDA code
├── cuda/
│   └── main.cu        # CUDA kernel code
└── src/
    ├── cuda.rs        # Module to include compiled CUDA code
    └── main.rs        # Rust application code
```

## How It Works

1. The `build.rs` script compiles the CUDA kernel in `cuda/main.cu` to a fatbin file during the build process.
2. The `cuda.rs` module includes the compiled fatbin as a byte array.
3. The `main.rs` file loads the CUDA module, allocates memory, and launches the kernel.

## Running the Example

```bash
cargo run --release
```

## Modifying the Template

### Adding New CUDA Kernels

1. Edit `cuda/main.cu` to add your kernel functions.
2. Make sure to use `extern "C" __global__` for kernel functions to make them accessible to Rust.

### Using Different Kernel Functions

1. In `src/main.rs`, change the kernel function name when getting the function handle:
   ```rust
   let my_kernel = module.get_function("my_kernel_name")?;
   ```

2. Update the kernel launch parameters as needed:
   ```rust
   launch!(
       my_kernel<<<grid_size, block_size, 0, stream>>>(
           // Your parameters here
       )
   )?;
   ```

## License

MIT 