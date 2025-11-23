# track-splitter
Goal: Splits wave files into separate tracks.

So far: Calculates RMS values using SIMD and parallelism and finds all the gaps in an i32 PCM stereo wav file. I've been testing it on a 42 hour wav file and it takes about 500 ms to calculate the RMS level. The silence finder says my file has a 302 minute silence and a 177 minute silence.

This is my first Rust project! :-)
