import librosa

def convert_wav(file, rate=16000):
    signal, sr = librosa.load(file, sr=None)
    new_signal = librosa.resample(signal, sr, rate)  #
    out_path = file.split('.wav')[0] + "_new.wav"
    librosa.output.write_wav(out_path, new_signal, rate)

    return out_path


file = "E:/sycl/emotion analyze/test/database/fear/10.wav"

print(convert_wav(file))
