from speechbrain.pretrained import EncoderDecoderASR

# load stt model
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")

# speech to text
text = asr_model.transcribe_file("example.wav")
print(text)