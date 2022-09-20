# mix_two_sound

2 つの音声を重畳させるスクリプトと、CSJ を使って、2 音声の重畳させるスクリプトの作成

# 重畳方法

以下のサイトを参考に組みました。

[任意の Signal-to-Noise 比の音声波形を Python で作ろう！](https://engineering.linecorp.com/ja/blog/voice-waveform-arbitrary-signal-to-noise-ratio-python/)

# CSJ の重畳コーパスの作り方

1. まず CSJ を、プロジェクトルート直下に配置します。

2. CJS＿mix スクリプトを叩く

    ```bash
    python CSJ_mix.py --csj_path ** --dist_path CSJ_mix
    ```

3. Comming soon
