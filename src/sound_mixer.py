from copyreg import remove_extension
import logging
import numpy as np
from typing import Tuple
from scipy.io.wavfile import write
from pathlib import Path
import os
import soundfile as sf


class TwoSoundSuperimposition:
    @staticmethod
    def cal_amp(wf):
        """
        calculate amplitude
        """
        buffer = wf.readframes(wf.getnframes())
        amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
        return amptitude

    @staticmethod
    def cal_rms(amp: np.ndarray) -> float:
        """
        calculate Root Mean Square
        """
        mean = np.mean(np.square(amp))
        return np.sqrt(mean)

    @staticmethod
    def cal_adjusted_rms(clean_rms: float, snr: float) -> float:
        """
        calculate adjusted Root Mean Square
        """
        a = snr / 20
        noise_rms = clean_rms / (10**a)
        return noise_rms

    @staticmethod
    def remove_extension(path:str)->str:
        """
        get path string removed file extension
        """
        target="."
        idx = path.rfind(target)
        return path[:idx]  # スライスで半角空白文字よりも前を抽出

    @classmethod
    def superimposition(
        self, sound1_file: str, sound2_file: str, snr: float, dist_dir: str, write_ref:bool = False,filename:str=""
    ) -> Tuple[str, str, str]:
        """
        snrの値で2音声を重畳する
        sound1/sound2 = snr
        Args:
            sound1_file(str): 重畳する1つ目の音声
            sound2_file(str): 重畳する2つ目の音声
            snr(float): Speech Noise Ratio
        Return:
            str: 重畳音声のID
            str: 重畳済みの音声ファイル
            str: 重畳前のsound1
            str: 重畳前のsound2
        """

        # wav限定の処理
        """
        # sound 1
        with wave.open(sound1_file, "r") as f:
            sound1_data = self.cal_amp(f)
            if sound1_data.ndim == 2:
                sound1_data = sound1_data[0]
            sound1_rate = f.getframerate()
        
        # sound 2
        with wave.open(sound2_file, "r") as f:
            sound2_data = self.cal_amp(f)
            if sound2_data.ndim == 2:
                sound2_data = sound2_data[0]
            sound2_rate = f.getframerate()
        """
        sound1_data, sound1_rate = sf.read(sound1_file)
        sound2_data, sound2_rate = sf.read(sound2_file)

        sound1_rms = self.cal_rms(sound1_data)
        sound2_rms = self.cal_rms(sound2_data)

        # SNRに対応したRMSを求める
        if sound2_rms!=0:
            adjusted_sound2_rms = self.cal_adjusted_rms(sound1_rms, snr)
            adjusted_sound2_data = sound2_data * (adjusted_sound2_rms / sound2_rms)
        else:
            adjusted_sound2_data = sound2_data
        adjusted_sound2_data = adjusted_sound2_data.astype(np.float64)

        # 長さ調整(0 padding)
        sound1_len = len(sound1_data)
        sound2_len = len(adjusted_sound2_data)
        if sound1_len < sound2_len:
            sound1_data = np.pad(sound1_data, (0, sound2_len - sound1_len))
        elif sound2_len < sound1_len:
            adjusted_sound2_data = np.pad(
                adjusted_sound2_data, (0, sound1_len - sound2_len)
            )

        # 重畳
        mix_data = sound1_data + adjusted_sound2_data
        if sound1_rate != sound2_rate:
            logging.error("sampling rate is different")
            return
        mix_rate = sound1_rate

        # waveパッケージ特有の処理
        """
        # 正規化 (wavが16bitなので、符号をどけた2^15 ~ -2^15の値に正規化)
        max_value = np.abs(mix_data).max()
        if max_value > 32767:
            mix_data = mix_data * (32767 / max_value)
            adjusted_sound2_data = adjusted_sound2_data * (32767 / max_value)
            sound1_data = sound1_data * (32767 / max_value)

        # dtype int16に変換
        sound1_data = np.asarray(sound1_data.astype(np.int16))
        adjusted_sound2_data = np.asarray(adjusted_sound2_data.astype(np.int16))
        mix_data = np.asarray(mix_data.astype(np.int16))
        """

        #mixed_soundの名前生成のために，ファイル名から拡張子を除いたものを取得
        sound1_name = self.remove_extension(Path(sound1_file).name)
        sound2_name = self.remove_extension(Path(sound2_file).name)
        

        # distファイルの存在チェック
        if not Path(dist_dir).exists():
            os.makedirs( Path(dist_dir),exist_ok=True)
        if (write_ref):
            if not (Path(dist_dir) / "sound1").exists():
                os.makedirs((Path(dist_dir) / "sound1"),exist_ok=True)
            if not (Path(dist_dir) / "sound2").exists():
                os.makedirs((Path(dist_dir) / "sound2"),exist_ok=True)
            if not (Path(dist_dir) / "mix").exists():
                os.makedirs((Path(dist_dir) / "mix"),exist_ok=True)

        # write sound1 and sound2
        if (write_ref):
            sound1_path = Path(dist_dir) / "sound1" / f"{sound1_name}.wav"
            sound2_path = Path(dist_dir) / "sound2" / f"{sound2_name}.wav"
            sf.write(str(sound1_path), sound1_data, sound1_rate)
            sf.write(str(sound2_path), adjusted_sound2_data, sound2_rate)

        if filename == "":
            mixname=f"{sound1_name}_{sound2_name}.wav"
        else:
            mixname=filename

        
        # write mix sound
        if (write_ref):
            mix_path = Path(dist_dir) / "mix" / mixname
        else:
            mix_path = Path(dist_dir)/ mixname
        # write(mix_path, mix_rate, mix_data)
        sf.write(str(mix_path),mix_data,mix_rate)

        mix_id = f"{sound1_name}_{sound2_name}"
        #return mix_id, mix_path, sound1_path, sound2_path
        return mix_id, mix_path, sound1_file, sound2_file


# test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename="sound_mixer_debug.log")
    # sound1 = "sample_data/sound1.wav"
    sound2 = "sample_data/sound2.wav"
    sound1="/mnt/data1/matsumoto/dump_wav2vec2_2/raw/eval2/data/format.17/A02M0012_0472185_0480985.flac"
    snr = 0.0
    dist = "sample_data/result"
    TwoSoundSuperimposition.superimposition(sound1, sound2, snr, dist)
