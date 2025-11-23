//use std::slice;
#![feature(portable_simd)]

use std::cmp::Ordering;
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

fn rms(sum_squares : &(f64x32, u64x32)) -> f64 {
    (sum_squares.0.reduce_sum() / (sum_squares.1.cast() as f64x32).reduce_sum()).sqrt()
}

pub struct SilenceIter<'a, T> {
    data: &'a[T],
    current: Option<&'a[T]>,
    empty: bool,
}

impl<'a, T: std::cmp::PartialEq<i32x32>> Iterator for SilenceIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<&'a [T]> {
        if self.empty {
            return None;
        }
        let search_start : usize = if self.current.is_none() {
            0
        } else {
            if self.current?.as_ptr() as usize + self.current?.len()*128 == self.data.as_ptr() as usize + self.data.len()*128 {
                self.current = None;
                return self.current;
            }
            // initialize to the item past the end of the current silent section
            (self.current?.as_ptr() as usize - self.data.as_ptr() as usize + self.current?.len()*128) / 128
        };

        const ZERO: i32x32 = Simd::splat(0_i32);
        let mut i = search_start;
        while i < self.data.len() {
            if self.data[i] == ZERO {
                let silence_start = i;
                while i < self.data.len() && self.data[i] == ZERO {
                    i += 1;
                };
                self.current = Option::from(&self.data[silence_start..i]);
                return self.current;
            }
            i += 1;
        }
        if self.current.is_none() {
            self.empty = true;
        }
        self.current = None;
        return self.current;
    }
}

impl<'a, T: std::cmp::PartialEq<i32x32>> DoubleEndedIterator for SilenceIter<'a, T> {
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.empty {
            return None;
        }
        let search_start : usize = if self.current.is_none() {
            self.data.len()/128 - 1
        } else {
            if self.current?.as_ptr() == self.data.as_ptr() {
                self.current = None;
                return self.current;
            }
            // initialize to the item before the beginning of the current silent section
            (self.current?.as_ptr() as usize - self.data.as_ptr() as usize) / 128 - 1
        };

        const ZERO: i32x32 = Simd::splat(0_i32);
        let mut i = search_start;
        while i > 0 {
            if self.data[i] == ZERO {
                let silence_end = i;
                while i > 0 && self.data[i] == ZERO {
                    i -= 1;
                };
                if self.data[i] != ZERO {
                    self.current = Option::from(&self.data[i + 1..silence_end + 1]);
                    return self.current;
                } else {
                    self.current = Option::from(&self.data[i..silence_end + 1]);
                    return self.current;
                }
            }
            i -= 1;
        }
        if self.data[i] != ZERO {
            if self.current.is_none() {
                self.empty = true;
            }
            self.current = None;
            return self.current;
        } else {
            self.current = Option::from(&self.data[0..1]);
            return self.current;
        }
    }
}

struct SilenceHeapValue<'a>(&'a [i32x32]);

impl<'a> SilenceHeapValue<'a> {
    fn new(value: &'a[i32x32]) -> Self {
        SilenceHeapValue(value)
    }

    fn get_inner_value(&self) -> &'a [i32x32] {
        &self.0
    }
}

impl<'a> PartialEq for SilenceHeapValue<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
    }
}

impl<'a> Eq for SilenceHeapValue<'a> {}

impl<'a> PartialOrd for SilenceHeapValue<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.0.len().partial_cmp(&other.0.len()) {
            Some(Ordering::Equal) => {
                self.0.as_ptr().partial_cmp(&other.0.as_ptr())
            }
            other_ordering => other_ordering,
        }
    }
}

impl<'a> Ord for SilenceHeapValue<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(other.0).unwrap()
    }
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
    let sample_count = wav_file.mmap.len() / 8; // i32 pcm stereo

    //println!("Squares: {:?}", squares);
    println!("RMS value: {:?} over {}:{:02}:{:02} calculated in {} ms", rms, sample_count / 48000 / 3600, (sample_count / 48000 / 60) % 60, (sample_count / 48000) % 60, elapsed_time.as_millis());

    let silence_iter = SilenceIter { data: wav_file.get_data_i32(), current: None, empty: false };
    let mut silence_heap : BinaryHeap<SilenceHeapValue> = BinaryHeap::new();
    for silence in silence_iter {
        silence_heap.push(SilenceHeapValue::new(&silence));
    }

    let biggest_silences: Vec<_> = silence_heap.iter().take(10).collect();
    for silence_wrapper in biggest_silences {
        let silence = silence_wrapper.get_inner_value();
        let silence_begin = (silence.as_ptr() as usize - wav_file.get_data_i32().as_ptr() as usize) / 128;

        println!("Silence: at {:?} minutes and {}:{:02}:{:03} mm:ss:ms long", silence_begin*16 / 48000 / 60,
                 silence.len()*16 / 48000 / 60, (silence.len()*16 / 48000)%60, (silence.len()*16 / 48)%1000);
    }
    println!("Number of silences: {}", silence_heap.len());
    Ok(())
}
