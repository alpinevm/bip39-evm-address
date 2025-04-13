mod cuda;
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
