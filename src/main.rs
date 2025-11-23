//use std::slice;
#![feature(portable_simd)]
use std::env;
use std::fs::File;
use std::io;
use std::simd::f64x32;
use std::simd::u64x32;
use std::simd::i32x32;
use std::simd::prelude::*;
use memmap2::MmapOptions;
use memmap2::Mmap;
use rayon::prelude::*;
use std::time::Instant;
use std::collections::BinaryHeap;

fn is_valid_wav_header(header: &[u8]) -> bool {
    header.len() >= 12 &&
        &header[0..4] == b"RIFF" &&
        &header[8..12] == b"WAVE"
}

struct WavFile<T> {
    pub mmap: T,
}

impl WavFile<Mmap> {
    pub fn open(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file) }?;

        let header = &mmap[0..12];
        if !is_valid_wav_header(header) {
            return Err(io::Error::new(io::ErrorKind::Other, "Bad WAVE header"));
        }
        println!("Wave file size: {}", mmap.len());
        Ok(WavFile { mmap })
    }

    fn get_data_i32(&self) -> &[i32x32] {
        let modulo = self.mmap[44..].as_ptr() as usize % 128;
        if modulo == 0 {
            return unsafe { std::slice::from_raw_parts(self.mmap[44..].as_ptr() as *const i32x32, self.mmap[44..].len() / 128) }
        }

        unsafe {
            std::slice::from_raw_parts(self.mmap[44 + 128 - modulo..].as_ptr() as *const i32x32, self.mmap[44 + 128 - modulo..].len() / 128)
        }
    }
}

pub trait SumSquares {
    fn sum_squares(&self) -> (f64x32, u64x32);
}

impl SumSquares for &[i32x32]
where
    std::simd::LaneCount<32>: std::simd::SupportedLaneCount,
    f64: std::simd::SimdElement,
    std::simd::Simd<i32, 32>: Sync,
{
    fn sum_squares(&self) -> (f64x32, u64x32) {
        self.par_iter().map(|a| ((a.cast() / Simd::splat(2147483648.0_f64)) * (a.cast() / Simd::splat(2147483648.0_f64)), Simd::splat(1_u64)))
            .reduce(|| (Simd::splat(0.0_f64), Simd::splat(0_u64)),
                    |a, b| (a.0 + b.0, a.1 + b.1))
    }
}

    /*
        fn sum_squares_heap(&self, heap: &mut BinaryHeap<f64>, n: usize) {

            assert_eq!(n % 16, 0);
            let heap_count = (self.len() / (n/16)).next_power_of_two() * 2 - 1;
            heap.drain();
            heap.reserve_exact(heap_count);
    */

fn rms(sum_squares : &(f64x32, u64x32)) -> f64 {
    (sum_squares.0.reduce_sum() / (sum_squares.1.cast() as f64x32).reduce_sum()).sqrt()
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        println!("{} <wav file path>", args[0]);
        return Ok(());
    }
    let wav_file = WavFile::open(&args[1]).expect("Failed to open file");

    let start_time = Instant::now();
    let squares = wav_file.get_data_i32().sum_squares();
    let rms = rms(&squares);
    let elapsed_time = start_time.elapsed();
    let sample_count = wav_file.mmap.len() / 8;

    //println!("Squares: {:?}", squares);
    println!("RMS value: {:?} over {}:{:02}:{:02} calculated in {} ms", rms, sample_count / 48000 / 3600, (sample_count / 48000 / 60) % 60, (sample_count / 48000) % 60, elapsed_time.as_millis());
    Ok(())
}
