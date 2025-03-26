import librosa
import numpy as np
import onnxruntime as ort_dml
import soundfile

import logging

logger = logging.getLogger(__name__)


class ContentVec:

    # v1 models use: vec-256-layer-9,     v2 models use: vec-768-layer-12
    def __init__(self, vec_path="vec-768-layer-12.onnx", device=None):
        logger.info("Load model(s) from {}".format(vec_path))
        if device == "cpu" or device is None:
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "dml":
            providers = ["CPUExecutionProvider"]
        else:
            raise RuntimeError("Unsupported Device")
        self.model = ort_dml.InferenceSession(vec_path, providers=providers)

    def __call__(self, wav):
        return self.forward(wav)

    def forward(self, wav):
        feats = wav
        if feats.ndim == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.ndim == 1, feats.ndim
        feats = np.expand_dims(np.expand_dims(feats, 0), 0)
        onnx_input = {self.model.get_inputs()[0].name: feats}
        logits = self.model.run(None, onnx_input)[0]
        return logits.transpose(0, 2, 1)


def get_f0_predictor(f0_predictor, hop_length, sampling_rate, **kargs):
    if f0_predictor == "pm":
        from tts_with_rvc.lib.infer_pack.modules.F0Predictor.PMF0Predictor import PMF0Predictor

        f0_predictor_object = PMF0Predictor(
            sampling_rate=sampling_rate
        )
    elif f0_predictor == "harvest":
        from tts_with_rvc.lib.infer_pack.modules.F0Predictor.HarvestF0Predictor import (
            HarvestF0Predictor,
        )

        f0_predictor_object = HarvestF0Predictor(
            hop_length=hop_length, sampling_rate=sampling_rate
        )
    elif f0_predictor == "dio":
        from tts_with_rvc.lib.infer_pack.modules.F0Predictor.DioF0Predictor import DioF0Predictor

        f0_predictor_object = DioF0Predictor(
            hop_length=hop_length, sampling_rate=sampling_rate
        )
    else:
        raise Exception("Unknown f0 predictor")
    return f0_predictor_object


class OnnxRVC:
    def __init__(
        self,
        model_path,
        sr=40000,
        hop_size=512,
        vec_path="vec-768-layer-12.onnx",
        device="cpu",
    ):
        self.vec_model = ContentVec(vec_path, device)
        if device == "cpu" or device is None:
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "dml":
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            raise RuntimeError("Unsportted Device")
        self.model = ort_dml.InferenceSession(model_path, providers=providers)
        self.sampling_rate = sr
        self.hop_size = hop_size

    def forward(self, hubert, hubert_length, pitch, pitchf, ds, rnd):
        onnx_input = {
            self.model.get_inputs()[0].name: hubert,
            self.model.get_inputs()[1].name: hubert_length,
            self.model.get_inputs()[2].name: pitch,
            self.model.get_inputs()[3].name: pitchf,
            self.model.get_inputs()[4].name: ds,
            self.model.get_inputs()[5].name: rnd,
        }
        return (self.model.run(None, onnx_input)[0] * 32767).astype(np.int16)

    def segment_audio(self, wav, sr):
        segment_length = sr * 50  # Maximum segment length of 50 seconds
        lookback_window = sr * 10  # Lookback window for 10 seconds

        segments = []
        start_idx = 0
        iteration = 0  # Debug

        while start_idx < len(wav):
            end_idx = min(start_idx + segment_length, len(wav))
            
            if start_idx == end_idx:  # Check if the segment duration is 0
                print(f"Segment duration is 0 at start_idx {start_idx}, end_idx {end_idx}")
                break

            segment = wav[start_idx:end_idx]

            # Check if it's the last segment
            if end_idx == len(wav):
                segments.append(segment)
                break
            
            # Look back for 10 seconds within the 50-second interval
            lookback_start = max(start_idx - lookback_window, 0)
            segment_lookback = wav[lookback_start:start_idx]

            # Calculate RMS for the lookback window
            rms_lookback = librosa.feature.rms(y=segment_lookback, frame_length=int(0.02 * sr), hop_length=int(0.02 * sr))[0]
            avg_rms_lookback = np.mean(rms_lookback)

            # Calculate RMS for the current segment
            rms_segment = librosa.feature.rms(y=segment, frame_length=int(0.02 * sr), hop_length=int(0.02 * sr))[0]
            avg_rms_segment = np.mean(rms_segment)

            if avg_rms_lookback < avg_rms_segment:
                min_rms_index = np.argmin(rms_lookback)
                new_end_idx = min(start_idx + segment_length, lookback_start + min_rms_index * int(0.02 * sr))
                if new_end_idx > start_idx:
                    end_idx = new_end_idx
                    print(f"Adjusted end_idx based on RMS: {end_idx}")

            segments.append(wav[start_idx:end_idx])
            print(f"Start Index: {start_idx}, End Index: {end_idx}")  # Debug
            print(f"Processed segment: {len(wav[start_idx:end_idx]) / sr} seconds")  # Debug
            start_idx = end_idx
            iteration += 1  # Debug

        # Adjust the last segment if it exceeds 50 seconds
        last_segment_duration = len(segments[-1]) / sr
        if last_segment_duration > 50.0:
            excess_samples = int((last_segment_duration - 50.0) * sr)
            segments[-1] = segments[-1][:len(segments[-1]) - excess_samples]

        return segments

    def inference(
        self,
        raw_path,
        sid,
        f0_method="dio",
        f0_up_key=0,
        pad_time=0.5,
        cr_threshold=0.02,
    ):
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0_predictor = get_f0_predictor(
            f0_method,
            hop_length=self.hop_size,
            sampling_rate=self.sampling_rate,
            threshold=cr_threshold,
        )
        # Loading the input audio
        wav, sr = librosa.load(raw_path, sr=self.sampling_rate)
        org_length = len(wav)
        if org_length / sr > 50.0:
            print(" Above 50sec audio detected. Splitting initiated ")
            segments = self.segment_audio(wav, sr)
        else:
            segments = [wav]
            
        print(f"Total segments: {len(segments)}")


        # Resample all segments at once
        segments_resampled = [librosa.resample(seg, orig_sr=sr, target_sr=16000) for seg in segments]
        
        processed_segments = []
        # cache gt_wav and resampled to 16khz wav segments
        for idx, (seg, seg_resampled) in enumerate(zip(segments, segments_resampled)):
            print(f"Processing segment {idx + 1} of {len(segments)}")
            
            # Apply existing script logic to each segment individually
            hubert = self.vec_model(seg_resampled)
            
            hubert = np.repeat(hubert, 2, axis=2).transpose(0, 2, 1).astype(np.float32)
            hubert_length = hubert.shape[1]

            pitchf = f0_predictor.compute_f0(seg, hubert_length)
            pitchf = pitchf * 2 ** (f0_up_key / 12)
            pitch = pitchf.copy()
            
            f0_mel = 1127 * np.log(1 + pitch / 700)
            f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
                f0_mel_max - f0_mel_min
            ) + 1
            f0_mel[f0_mel <= 1] = 1
            f0_mel[f0_mel > 255] = 255
            pitch = np.rint(f0_mel).astype(np.int64)


            pitchf = pitchf.reshape(1, len(pitchf)).astype(np.float32)
            pitch = pitch.reshape(1, len(pitch))
            ds = np.array([sid]).astype(np.int64)
        
            rnd = np.random.randn(1, 192, hubert_length).astype(np.float32)
            hubert_length = np.array([hubert_length]).astype(np.int64)

            out_wav_segment = self.forward(hubert, hubert_length, pitch, pitchf, ds, rnd).squeeze()
            out_wav_segment = np.pad(out_wav_segment, (0, 2 * self.hop_size), "constant")
            
            processed_segments.append(out_wav_segment)
            
        final_output = np.concatenate(processed_segments)
        
        return final_output[0:org_length]
