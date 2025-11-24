#![feature(portable_simd)]
#![feature(binary_heap_into_iter_sorted)]

use std::cmp::Ordering;
use std::{env, fs};
use std::fs::File;
use std::io;
use std::io::Write;
use std::simd::f64x32;
use std::simd::u64x32;
use std::simd::i32x32;
use std::simd::prelude::*;
use memmap2::MmapOptions;
use memmap2::Mmap;
use rayon::prelude::*;
use mp3lame_encoder::{Builder, /*Id3Tag,*/ InterleavedPcm, FlushNoGap};
use std::thread::scope;
use std::sync::mpsc;

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

pub trait IsZero {
    fn is_zero(&self) -> bool;
}

impl IsZero for i32x32
//where
//    std::simd::LaneCount<32>: std::simd::SupportedLaneCount,
//    f64: std::simd::SimdElement,
//    std::simd::Simd<i32, 32>: Sync,
{
    fn is_zero(&self) -> bool {
        const THRESHOLD: i32 = 0; // 0 = Look only for digital zero values. Works best with a digital recording.
        // Try using a larger value if you have an analog recording. These are i32 values so they should be quite
        // large compared to if they were i16 values.
        const MIN: i32x32 = Simd::splat(-THRESHOLD);
        const MAX: i32x32 = Simd::splat(THRESHOLD);

        if self > &MAX {
            return false;
        } else if self < &MIN {
            return false;
        }
        true
    }
}

pub struct SilenceIter<'a> {
    data: &'a[i32x32], // reference to entire file
    current: Option<&'a [i32x32]>, // reference to slice containing a silence period
    empty: bool, // true if there are no silence periods
}

impl<'a> Iterator for SilenceIter<'a> {
    type Item = &'a [i32x32];

    fn next(&mut self) -> Option<&'a [i32x32]> {
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

        let mut i = search_start;
        while i < self.data.len() {
            if self.data[i].is_zero() {
                let silence_start = i;
                while i < self.data.len() && self.data[i].is_zero() {
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

#[derive(Clone)]
struct SilenceHeapValue<'a>(&'a [i32x32]);

impl<'a> SilenceHeapValue<'a> {
    fn new(value: &'a[i32x32]) -> Self {
        SilenceHeapValue(value)
    }

    fn get_inner_value(&self) -> &'a [i32x32] {
        &self.0
    }

    fn gap(&self, prev : SilenceHeapValue) -> &'a [i32x32] {
        let prev_end = prev.get_inner_value().as_ptr() as usize + prev.get_inner_value().len() * 128;
        unsafe {
            std::slice::from_raw_parts(prev_end as *const i32x32, (self.get_inner_value().as_ptr() as usize - prev_end) / 128)
        }
    }

    pub const GAPSAMPLES : usize = 48000/16; // 62.5 ms
    pub const TRACKSAMPLES : usize = 48000 * 25;
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

fn write_mp3(name: String, data: &[i32x32]) {
    let mut mp3_encoder = Builder::new().expect("Create LAME builder");
    mp3_encoder.set_num_channels(2).expect("set channels");
    mp3_encoder.set_sample_rate(48_000).expect("set sample rate");
//    mp3_encoder.set
    mp3_encoder.set_brate(mp3lame_encoder::Bitrate::Kbps192).expect("set brate");
    mp3_encoder.set_quality(mp3lame_encoder::Quality::Best).expect("set quality");
    /*    mp3_encoder.set_id3_tag(Id3Tag {
            title: b"My title",
            artist: &[],
            album: b"My album",
            album_art: &[],
            year: b"Current year",
            comment: b"Just my comment",
        });*/
    let mut mp3_encoder = mp3_encoder.build().expect("To initialize LAME encoder");
    let mut mp3_out_buffer = Vec::new();

    //let mut encoded_size : usize = 0;
    for block in data {
        let array = block.to_array();

        let input = InterleavedPcm(&array);
        mp3_out_buffer.reserve(mp3lame_encoder::max_required_buffer_size(array.len()/2));
        let encoded_size = mp3_encoder.encode(input, mp3_out_buffer.spare_capacity_mut()).expect("To encode");
        unsafe {
            mp3_out_buffer.set_len(mp3_out_buffer.len().wrapping_add(encoded_size));
        }
    }

    let encoded_size = mp3_encoder.flush::<FlushNoGap>(mp3_out_buffer.spare_capacity_mut()).expect("to flush");
    unsafe {
        mp3_out_buffer.set_len(mp3_out_buffer.len().wrapping_add(encoded_size));
    }
    fs::write(name, mp3_out_buffer).expect("Failed to write MP3 data");
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        println!("{} <wav file path>", args[0]);
        return Ok(());
    }
    let wav_file = WavFile::open(&args[1]).expect("Failed to open file");

    let silence_iter = SilenceIter { data: wav_file.get_data_i32(), current: None, empty: false };
    let mut last_silence : Option<SilenceHeapValue> = None;
    let mut tracks : Vec<&[i32x32]> = Vec::new();
    for silence in silence_iter {
        // FIXME: get rid of clones
        let wrapped_silence = SilenceHeapValue::new(silence);
        if !last_silence.is_none() {
            let loud_samples = wrapped_silence.gap(last_silence.clone().unwrap());
            if wrapped_silence.get_inner_value().len()*16 >= SilenceHeapValue::GAPSAMPLES && loud_samples.len()*16 >= SilenceHeapValue::TRACKSAMPLES {
                last_silence = Option::from(wrapped_silence.clone());
                tracks.push(loud_samples);
            }
        } else {
            last_silence = Option::from(wrapped_silence.clone());
        }
    }
    let track_count = tracks.len();
    println!("Number of tracks? {}\n", track_count);

    let it = tracks.into_par_iter().enumerate();

    scope(|s| {
        let (tx, rx) = mpsc::channel();

        s.spawn(move || {
            it.for_each(|track: (usize, &[i32x32])| {
                write_mp3(format!("tracks/{:05}.mp3", track.0), track.1);
                tx.send(true).unwrap();
            });
        });

        for i in 0..track_count {
            print!("\rWriting track {}/{}...", i, track_count);
            io::stdout().flush().unwrap();
            let _ = rx.recv().unwrap();
        }
    });
    println!("\nDone!");

    Ok(())
}
