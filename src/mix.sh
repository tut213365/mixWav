. ~/wav2vec2_exp/espnet/tools/activate_python.sh

# ref_path=/home/katsuaki/WAV
ref_path=/mnt/data1/matsumoto/dump_wav2vec2_2/raw
noise_path=/mnt/data1/dataset/CHiME-3/CHiME3/data/audio/16kHz/backgrounds
dest_path=/home/katsuaki/MIXED_WAV
# dest_path=/home/katsuaki/mix_two_sound/sample_data/result
min_snr=0
max_snr=20.0

python3 /home/katsuaki/mix_two_sound/src/mymixer.py\
    --ref_dir "${ref_path}"\
    --noise_dir "${noise_path}"\
    --dest_dir "${dest_path}"\
    --min_snr "${min_snr}"\
    --max_snr "${max_snr}"\
    --show_ref false\
