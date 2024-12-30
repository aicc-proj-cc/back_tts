# tts_service.py
import numpy as np
import torch
from torch import no_grad, LongTensor
import scipy.io.wavfile as wavfile
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
from text import text_to_sequence
import commons

# 언어 마크 설정
language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "한국어": "[KO]",
    "Mix": "",
}

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# TTS 서비스 클래스
class TTSService:
    def __init__(self, config_path, model_checkpoint):
        # 하이퍼파라미터 로드
        self.hps = utils.get_hparams_from_file(config_path)
        
        # 모델 초기화
        self.model = SynthesizerTrn(
            len(self.hps.symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        ).to(device)
        self.model.eval()
        
        # 체크포인트 로드
        utils.load_checkpoint(model_checkpoint, self.model, None)
        
        # 스피커 ID 로드
        self.speaker_ids = self.hps.speakers

    def _get_text(self, text, is_symbol=False):
        text_norm = text_to_sequence(
            text, self.hps.symbols, [] if is_symbol else self.hps.data.text_cleaners
        )
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        return LongTensor(text_norm)

    def generate_tts(self, text, speaker, language, speed):
        print("generate_tts ::::", "text:", text, "speaker:", speaker, "language:",language, "speed:", speed)
        try:
            if language is not None:
                # text = language_marks[language] + text + language_marks[language]
                text = f"[{language}]{text}[{language}]"
            print("text ::::", text)
            speaker_id = self.speaker_ids[speaker]
            print("speaker_id ::::", speaker_id)
            stn_tst = self._get_text(text, is_symbol=False)

            with no_grad():
                x_tst = stn_tst.unsqueeze(0).to(device)
                x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
                sid = LongTensor([speaker_id]).to(device)

                audio = self.model.infer(
                    x_tst,
                    x_tst_lengths,
                    sid=sid,
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1.0 / speed,
                )[0][0, 0].data.cpu().float().numpy()

            return "Success", (self.hps.data.sampling_rate, audio)
        except Exception as e:
            return "Error", str(e)
