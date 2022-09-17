import logging
import numpy as np
import wave
from typing import Tuple
from scipy.io.wavfile import write
from pathlib import Path


class TwoSoundSuperimposition:
    def __init__(self):
        pass

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

    def superimposition(
        self, sound1_file: str, sound2_file: str, snr: float, dist_dir: str
    ) -> Tuple[str, str, str]:
        """
        snrの値で2音声を重畳する
        sound1/sound2 = snr
        Args:
            sound1_file(str): 重畳する1つ目の音声
            sound2_file(str): 重畳する2つ目の音声
            snr(float): Speech Noise Ratio
        Return:
            str: 重畳済みの音声ファイル
            str: 重畳前のsound1
            str: 重畳前のsound2
        """

        # sound 1
        with wave.open(sound1_file, "r") as f:
            sound1_data = self.cal_amp(f)
            if sound1_data.ndim == 2:
                sound1_data = sound1_data[0]
            sound1_rate = f.getframerate()
        # sound 1
        with wave.open(sound2_file, "r") as f:
            sound2_data = self.cal_amp(f)
            if sound2_data.ndim == 2:
                sound2_data = sound2_data[0]
            sound2_rate = f.getframerate()

        sound1_rms = self.cal_rms(sound1_data)
        sound2_rms = self.cal_rms(sound2_data)

        # SNRに対応したRMSを求める
        adjusted_sound2_rms = self.cal_adjusted_rms(sound1_rms, snr)
        adjusted_sound2_data = sound2_data * (adjusted_sound2_rms / sound2_rms)
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

        sound1_name = Path(sound1_file).name.split(".")[0]
        sound2_name = Path(sound2_file).name.split(".")[0]

        # distファイルの存在チェック
        if not Path(dist_dir).exists():
            Path(dist_dir).mkdir(exist_ok=True)
        if not (Path(dist_dir) / "sound1").exists():
            (Path(dist_dir) / "sound1").mkdir(exist_ok=True)
        if not (Path(dist_dir) / "sound2").exists():
            (Path(dist_dir) / "sound2").mkdir(exist_ok=True)
        if not (Path(dist_dir) / "mix").exists():
            (Path(dist_dir) / "mix").mkdir(exist_ok=True)

        # write sound1 and sound2
        sound1_path = Path(dist_dir) / "sound1" / f"{sound1_name}.wav"
        sound2_path = Path(dist_dir) / "sound2" / f"{sound2_name}.wav"
        write(sound1_path, sound1_rate, sound1_data)
        write(sound2_path, sound2_rate, adjusted_sound2_data)

        # write mix sound
        mix_path = Path(dist_dir) / "mix" / f"{sound1_name}_{sound2_name}.wav"
        write(mix_path, mix_rate, mix_data)

        return mix_path, sound1_path, sound2_path


# test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename="sound_mixer_debug.log")
    superimposition = TwoSoundSuperimposition()
    sound1 = "sample_data/sound1.wav"
    sound2 = "sample_data/sound2.wav"
    snr = 0.0
    dist = "sample_data/result"
    superimposition.superimposition(sound1, sound2, snr, dist)
