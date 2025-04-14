mod cuda;
mod utils;
use cuda::CUDA_FATBIN;
use cust::prelude::*;
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize CUDA
    println!("Initializing CUDA...");
    cust::init(CudaFlags::empty())?;

    let device = Device::get_device(0)?;
    println!("Using device: {}", device.name()?);

    let _ctx = Context::new(device)?;
    let module = Module::from_fatbin(CUDA_FATBIN, &[])?;
    let square_num = module.get_function("square_num")?;

    // Create input data
    let host_input: Vec<u32> = (0..100_000).collect();
    let host_input_count: u32 = host_input.len() as u32;

    // Allocate output buffer
    let mut host_output = vec![0u32; host_input.len()];

    // Allocate device memory
    let device_input = DeviceBuffer::<u32>::from_slice(&host_input)?;
    let device_output = DeviceBuffer::<u32>::from_slice(&mut host_output)?;

    // Configure kernel launch parameters
    let block_size: u32 = 1024;
    let grid_size: u32 = (host_input_count + block_size - 1) / block_size;
    println!(
        "Launching kernel with {} blocks of {} threads each",
        grid_size, block_size
    );

    // Launch the kernel
    unsafe {
        let time = Instant::now();
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;

        launch!(
            square_num<<<grid_size, block_size, 0, stream>>>(
                device_input.as_device_ptr(),
                host_input_count,
                device_output.as_device_ptr()
            )
        )?;
        stream.synchronize()?;
        println!(
            "Squared {} numbers in {:?}",
            host_input_count,
            time.elapsed()
        );
    }

    // Copy results back to host
    device_output.copy_to(&mut host_output)?;

    println!("First few squared numbers:");
    for idx in 0..5 {
        println!("Number {}^2: {}", idx, host_output[idx]);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keccak256_hash() {
        cust::init(CudaFlags::empty()).unwrap();

        let device = Device::get_device(0).unwrap();
        println!("Using device: {}", device.name().unwrap());

        let _ctx = Context::new(device).unwrap();
        let module = Module::from_fatbin(CUDA_FATBIN, &[]).unwrap();
        let keccak256_hash = module.get_function("keccak256_hash").unwrap();

        let host_input = b"Hello, world!";
        let mut host_output = [0u8; 32];
        let device_input = DeviceBuffer::<u8>::from_slice(host_input).unwrap();
        let device_output = DeviceBuffer::<u8>::from_slice(&mut host_output).unwrap();

        unsafe {
            let time = Instant::now();
            let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();

            launch!(
                keccak256_hash<<<1, 1, 0, stream>>>(
                    device_input.as_device_ptr(),
                    host_input.len() as u32,
                    device_output.as_device_ptr()
                )
            )
            .unwrap();
            stream.synchronize().unwrap();
            println!("Keccak256 hash in {:?}", time.elapsed());
        }

        device_output.copy_to(&mut host_output).unwrap();

        println!(
            "Keccak256 hash of \"Hello, world!\" is: {}",
            hex::encode(host_output)
        );

        let mut cpu_keccak_hash_output = [0u8; 32];
        utils::keccak_hash_in_place(host_input, &mut cpu_keccak_hash_output);
        println!(
            "Keccak256 hash of \"Hello, world!\" is: {}",
            hex::encode(cpu_keccak_hash_output)
        );

        assert_eq!(host_output, cpu_keccak_hash_output);
    }

    #[test]
    fn test_sha512_hash() {
        cust::init(CudaFlags::empty()).unwrap();

        let device = Device::get_device(0).unwrap();
        println!("Using device: {}", device.name().unwrap());

        let _ctx = Context::new(device).unwrap();
        let module = Module::from_fatbin(CUDA_FATBIN, &[]).unwrap();
        let sha512_hash = module.get_function("sha512_hash").unwrap();

        let host_input = b"";
        let mut host_output = [0u8; 64];
        let device_input = DeviceBuffer::<u8>::from_slice(host_input).unwrap();
        let device_output = DeviceBuffer::<u8>::from_slice(&mut host_output).unwrap();

        unsafe {
            let time = Instant::now();
            let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();

            launch!(
                sha512_hash<<<1, 1, 0, stream>>>(
                    device_input.as_device_ptr(),
                    host_input.len() as u32,
                    device_output.as_device_ptr()
                )
            )
            .unwrap();
            stream.synchronize().unwrap();
            println!("SHA512 hash in {:?}", time.elapsed());
        }

        device_output.copy_to(&mut host_output).unwrap();

        println!(
            "SHA512 hash of \"Hello, world!\" is: {}",
            hex::encode(host_output)
        );

        let mut cpu_sha512_hash_output = [0u8; 64];
        utils::sha512_hash_in_place(host_input, &mut cpu_sha512_hash_output);
        println!(
            "SHA512 hash of \"Hello, world!\" is: {}",
            hex::encode(cpu_sha512_hash_output)
        );

        assert_eq!(host_output, cpu_sha512_hash_output);
    }

    #[test]
    fn test_secp256k1_compute_pubkey() {
        cust::init(CudaFlags::empty()).unwrap();

        let device = Device::get_device(0).unwrap();
        println!("Using device: {}", device.name().unwrap());

        let _ctx = Context::new(device).unwrap();
        let module = Module::from_fatbin(CUDA_FATBIN, &[]).unwrap();
        let secp256k1_compute_pubkey = module.get_function("secp256k1_compute_pubkey").unwrap();

        let host_input: [u32; 8] = [
            0x1aaaaaa, 0xbeeee, 0xdeeea, 0xdeee, 0xbeee, 0xdeee, 0xbeee, 0xdeee,
        ];

        let mut host_output_px = [0u32; 8];
        let mut host_output_py = [0u32; 8];
        let device_input = DeviceBuffer::<u32>::from_slice(&host_input).unwrap();
        let device_output_px = DeviceBuffer::<u32>::from_slice(&mut host_output_px).unwrap();
        let device_output_py = DeviceBuffer::<u32>::from_slice(&mut host_output_py).unwrap();

        unsafe {
            let time = Instant::now();
            let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();

            launch!(
                secp256k1_compute_pubkey<<<1, 1, 0, stream>>>(
                    device_input.as_device_ptr(),
                    device_output_px.as_device_ptr(),
                    device_output_py.as_device_ptr()
                )
            )
            .unwrap();
            stream.synchronize().unwrap();
            println!("secp256k1_compute_pubkey in {:?}", time.elapsed());
        }

        device_output_px.copy_to(&mut host_output_px).unwrap();
        device_output_py.copy_to(&mut host_output_py).unwrap();

        let host_output_px_bytes: [u8; 32] = host_output_px
            .iter()
            .map(|x| x.to_le_bytes())
            .flatten()
            .collect::<Vec<u8>>()
            .try_into()
            .unwrap();

        let host_output_py_bytes: [u8; 32] = host_output_py
            .iter()
            .map(|x| x.to_le_bytes())
            .flatten()
            .collect::<Vec<u8>>()
            .try_into()
            .unwrap();

        println!(
            "[gpu] secp256k1_compute_pubkey data: \npx is: {}\npy is: {}",
            hex::encode(host_output_px_bytes),
            hex::encode(host_output_py_bytes)
        );

        let mut cpu_secp256k1_compute_pubkey_output_px = [0u8; 32];
        let mut cpu_secp256k1_compute_pubkey_output_py = [0u8; 32];
        utils::secp256k1_compute_pubkey_in_place(
            host_input,
            &mut cpu_secp256k1_compute_pubkey_output_px,
            &mut cpu_secp256k1_compute_pubkey_output_py,
        );

        println!(
            "[cpu] secp256k1_compute_pubkey data: \npx is: {}\npy is: {}",
            hex::encode(cpu_secp256k1_compute_pubkey_output_px),
            hex::encode(cpu_secp256k1_compute_pubkey_output_py)
        );

        assert_eq!(host_output_px_bytes, cpu_secp256k1_compute_pubkey_output_px);
        assert_eq!(host_output_py_bytes, cpu_secp256k1_compute_pubkey_output_py);
    }
}
