# track-splitter
Splits wav files into separate tracks and encodes them as mp3.

So far, it works only with i32 PCM stereo wav files, however it handles very large files, tested to 59 gigabytes. It encodes the mp3s in parallel using multiple cores.

It looks for silence between tracks which isn't very foolproof but it splits the 59 gig file into 412 tracks which is a lot better for an mp3 player than having one single file.

Future work: Use ML to identify track transitions to make this work better.

This was my first Rust project! :-)

BTW: I put the mp3 files on my Shokz OpenSwim Pro bone conduction headphones so I can listen to music using the integrated player without having to stream and run down my phone. It's fantastic, I love it.