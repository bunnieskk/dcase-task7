# Code is written by: Manu Harju 

import librosa
import numpy as np
import soundfile as sf

from pathlib import Path
from tqdm import tqdm




def main():
    duration = 4.0
    sr = 32000

    audio_dir = Path('audio')
    output_dir = Path('output')

    audio_fns = sorted(audio_dir.glob('*.wav'))
    output_dir.mkdir(exist_ok=True, parents=True)

    # signal length & hop size
    tgt_len = int(sr * duration)
    hop_len = tgt_len
    min_len = tgt_len // 2

    for fn in tqdm(audio_fns):
        x, _ = librosa.load(fn, sr=sr, mono=True)

        n_segments = (len(x) - tgt_len + hop_len - 1) // hop_len + 1
        
        for k in range(n_segments):
            start = k * hop_len

            x_seg = x[start:start + tgt_len]
            assert len(x_seg) > 0 

            # ignore too short segments (only for multiple segment audios)
            if n_segments > 1 and len(x_seg) < min_len:
                break

            if len(x_seg) < tgt_len:
                x_seg = np.pad(x_seg, (0, tgt_len - len(x_seg)))

            assert len(x_seg) == tgt_len

            sf.write(output_dir / f'{fn.stem}-{k:02d}.wav', x_seg, sr)




if __name__ == '__main__':
    main()
