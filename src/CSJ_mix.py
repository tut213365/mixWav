"""
python CSJ_mix.py --csj_path ** --dist_path CSJ_mix --log_level DEBUG --mode mix
"""

import argparse
import subprocess
import random
import logging
from sound_mixer import TwoSoundSuperimposition
import glob
import copy
from pathlib import Path
from joblib import Parallel, delayed
from typing import Dict
import shutil
from tqdm import tqdm
import re
from joblib import Parallel, delayed

log_type = {
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "WARN": logging.WARN,
    "ERROR": logging.ERROR,
}


class CSJMix:
    def __init__(self, csj_path: str, mode: str) -> None:
        self.TRN_DIR = Path(csj_path) / "TRN" / "Form2"
        self.WAV_DIR = Path(csj_path) / "WAV"
        self.DATA_DIR = Path(csj_path) / "DATA"
        if mode not in ["female", "mix"]:
            raise RuntimeError("please select mode from 'female', 'mix'")
        self.mode = mode

    def create_kaldi_dir(self, kaldi_path: Path):
        if kaldi_path.exists():
            shutil.rmtree(str(kaldi_path))
        kaldi_path.mkdir()
        (kaldi_path / "CSJ_wav").mkdir()
        (kaldi_path / "all").mkdir()

    def split_wav(self, kaldi_path: Path):
        """
        .trn fileの秒数にしたがって、音声を分割し、<kaldi_dir>/CSJ_wavの中に格納する
        """

        logging.info("stplit wav!")
        trn_list = glob.glob(str(self.TRN_DIR / "*" / "*.trn"))
        if len(trn_list) == 0:
            raise RuntimeError("length of trn_list is 0")
        text_list = []
        try:
            for trn in tqdm(trn_list):
                talk_id = Path(trn).name.split(".")[0]
                target_wav_list = glob.glob(str(self.WAV_DIR / "*" / f"{talk_id}.wav"))
                assert len(target_wav_list) == 1, "Multiple target wavs exist."
                target_wav = target_wav_list[0]
                output_dir = kaldi_path / "CSJ_wav" / talk_id
                output_dir.mkdir()
                with open(trn, encoding="shift_jis") as f:
                    lines = f.readlines()

                def split(l: str):
                    start_end = l.split(" ")[1]
                    text = l.split(" ")[2]
                    text = re.sub("[L:|<雑音>|<咳>|<息>|<笑>|<泣>|\n]", "", text)
                    start = float(start_end.split("-")[0])
                    end = float(start_end.split("-")[1])
                    subprocess.run(
                        # trim start dulation
                        [
                            "sox",
                            target_wav,
                            str(output_dir / f"{talk_id}-{start}-{end}.wav"),
                            "trim",
                            f"{start}",
                            f"{end-start}",
                        ]
                    )
                    return f"{talk_id}-{start}-{end} {text}\n"

                text_list += Parallel(n_jobs=-1)(delayed(split)(line) for line in lines)
        except Exception as e:
            raise RuntimeError(f"subprocess error: {e}")
        logging.info("split wav is finished")
        with open(kaldi_path / "all" / "text", "w") as f:
            f.writelines(sorted(text_list))

    def get_talk_info(self) -> Dict:

        # get speaker infos
        with open(self.DATA_DIR / "speaker_data.csv", encoding="shift_jis") as f:
            lines = f.readlines()
        # 講演者ID,性別,生年代,出生地,居住年数,居住年数（言語形成期）,父出身地,母出身地,最終学歴,備考,講演ID
        sex_info = {}
        for line in lines[1:]:
            infos = line.split(",")
            sex_info[infos[0]] = infos[1]
        del lines
        # get talk infos
        with open(self.DATA_DIR / "talk_data.csv", encoding="shift_jis") as f:
            lines = f.readlines()
        # 講演ID,講演者ID,記録票バージョン,聴き手人数,収録作業者人数,印象評定者ID,テープレコーダ,学会種・模擬講演のテーマ等,学会における講演形式,講演使用器材・配布資料,マイクにかかる息,録音環境,定常ノイズ源,室内の残響,会場拡声装置,突発的ノイズ発生時間・発生源,備考,たどたどしい,流暢な,単調な,表情ゆたかな,自信のある,自信の無い,優しい,落ち着いた,落ち着きのない,いらいらした,緊張した,リラックスした,大きい声,小さい声,かすれた声,裏返った声,こもった声,重厚な,軽薄な,若々しい,年寄じみた,元気のある,元気の無い,聞き取りやすい,聞き取りにくい,生意気な,尊大な,鼻にかかった,高い,低い,きっぱりした,速い,遅い,講演の自発性,難解な専門用語の多少,発話スピード,発音の明瞭さ,方言の多少,発話スタイル,読み上げ,朗読の流暢性,講演の準備,得手・不得手,講演経験,最終学歴,テストセット,分節音・韻律情報(コア),形態論情報,節単位情報,重要文選択,作業者による自由要約,講演者自身による自由要約,係り受け情報,談話境界情報,集合印象評定,単独印象評定,収録時の年齢
        talk_info = {}
        female_num = 0
        male_num = 0
        for line in lines[1:]:
            infos = line.split(",")
            # TODO 必要なら公演IDをフィルタリングする処理をここで入れる
            # ステレオは排除
            if infos[2] == "対話":
                continue
            talk_info[infos[0]] = {"spk": infos[1], "sex": sex_info[infos[1]]}
            if sex_info[infos[1]] == "男":
                male_num += 1
            elif sex_info[infos[1]] == "女":
                female_num += 1

        if self.mode == "female":
            talk_info = {
                key: value for key, value in talk_info.items() if value["sex"] == "女"
            }

        elif self.mode == "mix":
            new_talk_info = {}
            num = min(female_num, male_num)
            for key, value in talk_info.items():
                female = 0
                male = 0
                if value["sex"] == "男":
                    if male > num:
                        continue
                    male += 1
                elif value["sex"] == "女":
                    if female > num:
                        continue
                    female += 1
                new_talk_info[key] = value

        else:
            raise RuntimeError(f"{self.mode} is not implemented!")
        return talk_info

    def make_csj_mix(self, kaldi_path: str) -> None:
        """
        CSJ_mixのコーパスを作成する。
        """

        kaldi_path: Path = Path(kaldi_path)
        result_wav_path = kaldi_path / "mix_CSJ"
        # kaldi形式のデータが格納されるディレクトリの作成
        if not kaldi_path.exists():
            self.create_kaldi_dir(kaldi_path)
            self.split_wav(kaldi_path)

        sound1_list = []
        # 音声ファイルパスを準備
        talk_info = self.get_talk_info()
        for talk_key in talk_info.keys():
            search_path = kaldi_path / "CSJ_wav" / talk_key / f"*.wav"
            path_list = glob.glob(str(search_path))
            sound1_list += path_list

        sound2_list = copy.deepcopy(sound1_list)
        random.shuffle(sound2_list)

        logging.info("重畳開始")

        # 並列処理
        result_list = []
        for s1, s2 in zip(tqdm(sound1_list), sound2_list):
            result_list.append(
                TwoSoundSuperimposition.superimposition(
                    s1, s2, random.uniform(0, 5), result_wav_path
                )
            )
        logging.info("重畳終了")

        # kaldi形式に変換
        # sound1.scp, sound2.scp, mix.scp, sound1_text, sound2_text
        mix_scp = []
        sound1_scp = []
        sound2_scp = []
        for mix_id, mix, sound1, sound2 in result_list:
            mix_scp.append(f"{mix_id} {mix}\n")
            sound1_scp.append(f"{mix_id} {sound1}\n")
            sound2_scp.append(f"{mix_id} {sound2}\n")

        with open(kaldi_path / "all" / "mix.scp", "w") as f:
            f.writelines(mix_scp)
        with open(kaldi_path / "all" / "sound1.scp", "w") as f:
            f.writelines(sound1_scp)
        with open(kaldi_path / "all" / "sound2.scp", "w") as f:
            f.writelines(sound2_scp)
        logging.info("すべて終了")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("CSJからCSJ_mixを作成するスクリプト")
    parser.add_argument(
        "--csj_path",
        type=str,
    )
    parser.add_argument("--dist_path", type=str)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--log_level", type=str, default="INFO", choices=log_type.keys()
    )
    parser.add_argument("--mode", type=str, choices=["female", "mix"])
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(level=log_type[args.log_level])
    random.seed(args.seed)

    csj_mix = CSJMix(args.csj_path, args.mode)
    csj_mix.make_csj_mix(args.dist_path)


if __name__ == "__main__":
    main()
