from sound_mixer import TwoSoundSuperimposition
import logging
import argparse
import random
from distutils.util import strtobool
import glob
import re
import soundfile as sf
import wave
from pathlib import Path
from scipy.io.wavfile import write
import numpy as np
import os

debug=False


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("音声データにノイズを重畳するスクリプト")
    parser.add_argument("--ref_dir",type=str)
    parser.add_argument("--noise_dir",type=str)
    parser.add_argument("--dest_dir", type=str)
    parser.add_argument("--output_format", type=str,default="wav")
    parser.add_argument("--min_snr", type=float)
    parser.add_argument("--max_snr", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show_ref", type=str,default="false") 
    return parser


def mix():
    parser = get_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    logging.basicConfig(level=logging.DEBUG, filename="sound_mixer_debug.log")
    
    sound =args.ref_dir # "sample_data/sound1.wav"
    noise =args.noise_dir #"sample_data/sound2.wav"
    snr = args.max_snr*random.random() #0.0
    dest =args.dest_dir #"sample_data/result"
    show_ref=strtobool(args.show_ref)
    print(TwoSoundSuperimposition.superimposition(sound, noise, snr, dest,show_ref))      

def sound_extract(sound_path,dest_path,length,location:float=0.0):
    """
    extract part of data from sound_path into dest_path
    sound_path: path to wav to extract
    dest_path: path to write
    length: length of extracted data
    location: extract lacation in wav [0,1]
    """
    # 読み込み
    sound_data, sound_rate = sf.read(sound_path)
    sound_len=len(sound_data)

    """
    # 1/5の確率でノイズのないデータを作る
    if random.random()<0.2:
        sound_data*=0
    """

    if debug:
        print(sound_data[:10])

    # 長さlengthのデータ抽出
    size_ratio=int(length/sound_len)
    length=length%sound_len
    start=int(location*(sound_len-length-1))
    end=start+length
    extracted_data=[]
    if size_ratio>0:
        for i in range(size_ratio):
            extracted_data.append(sound_data)
    extracted_data.append(sound_data[start:end])
    # データ書き込み
    extracted_data=np.concatenate(extracted_data)
    extracted_data=np.asarray(extracted_data)
    
    if debug:
        print(extracted_data[:10])
    
    sf.write(str(dest_path),extracted_data,sound_rate)

    if debug:
        # print(extracted_data)
        print((length,len(extracted_data)))

    return
    
    
    
    
    



def main():
    if debug:
        print("getting argument...")
    parser = get_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    logging.basicConfig(level=logging.DEBUG, filename="sound_mixer_debug.log")

    TEMP_PATH="temp.wav"
    
    ref_dir =args.ref_dir # "sample_data/sound1.wav"
    noise_dir =args.noise_dir #"sample_data/sound2.wav"
    dest_dir =args.dest_dir #"sample_data/result"
    output_format=args.output_format
    min_snr = args.min_snr #0.0
    max_snr = args.max_snr #0.0
    show_ref=strtobool(args.show_ref)


    # ノイズと音声データのパスをglobで再帰的に取得
    input_format=["wav","flac"]
    ref_pathes=[]
    noise_pathes=[]
    for format in input_format:
        if debug:
            print("getting pathes at"+ref_dir+"/**/*."+format)
        ref_pathes.extend(glob.glob(ref_dir+"/**/*."+format,recursive=True))
        if debug:
            print("getting pathes at"+noise_dir+"/**/*."+format)
        noise_pathes.extend(glob.glob(noise_dir+"/**/*."+format,recursive=True))
        if debug:
            print(noise_pathes[0])
    noise_pathes_len=len(noise_pathes)

    if debug:
        print("mixing sounds...")
    
    if debug:
        ref_pathes=ref_pathes[:5]
    # 各音声データにノイズ重畳
    for ref_path in ref_pathes:
        if debug:
            print(ref_path)
        # 子ディレクトリ以下の構造は保持するようにdestination pathを設定
        child_path=re.search(ref_dir+r'(.*)',ref_path).group(1) 
        dest_path=dest_dir+child_path

        #重畳ノイズ生成
        noise_path=noise_pathes[random.randint(0,noise_pathes_len-1)]# noiseをランダム抽出
        if debug:
            print(noise_path)
        ref_data, ref_rate = sf.read(ref_path)
        ref_length=len(ref_data)
        sound_extract(noise_path,TEMP_PATH,ref_length,random.random()) # TEMP_PATHにrefと長さが同じnoiseを生成

        snr=max_snr*random.uniform(min_snr,max_snr)# snrを設定
        
        filename=os.path.split(dest_path)[1]# ファイル名
        dest_path=os.path.split(dest_path)[0]# フォルダ名
        filename=TwoSoundSuperimposition.remove_extension(filename)+"."+output_format
        print(filename)

        # ノイズ重畳
        logging.info(TwoSoundSuperimposition.superimposition(ref_path, TEMP_PATH, snr, dest_path,show_ref,filename))
        
    return



    


if __name__ == "__main__":
    main()
    
