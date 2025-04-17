# File: tts_with_rvc/lib/infer_pack/onnx_inference.py
import librosa
import numpy as np
import onnxruntime as ort
import soundfile
import logging
import os
import faiss
from scipy import signal
import torch
import torch.nn.functional as F
from scipy.signal import get_window
from librosa.util import pad_center
import sys
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# --- change_rms ---
def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)[0]
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)[0]
    rms1_torch = torch.from_numpy(rms1).float().unsqueeze(0).unsqueeze(0)
    rms2_torch = torch.from_numpy(rms2).float().unsqueeze(0).unsqueeze(0)
    target_len = data2.shape[0]
    rms1_interp = F.interpolate(rms1_torch, size=target_len, mode='linear', align_corners=False).squeeze()
    rms2_interp = F.interpolate(rms2_torch, size=target_len, mode='linear', align_corners=False).squeeze()
    rms2_interp = torch.max(rms2_interp, torch.tensor(1e-6))
    weight = (torch.pow(rms1_interp, 1 - rate) * torch.pow(rms2_interp, rate - 1))
    data2 *= weight.numpy()
    return data2

# --- MelSpectrogram ---
class MelSpectrogram(torch.nn.Module):
    def __init__(
        self, n_mel_channels, sampling_rate, win_length, hop_length,
        n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = librosa.filters.mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels,
            fmin=mel_fmin, fmax=mel_fmax, htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1, center=True):
        if isinstance(audio, np.ndarray): audio = torch.from_numpy(audio).float()
        elif not isinstance(audio, torch.Tensor): raise TypeError("Input audio must be a numpy array or torch tensor.")
        if audio.dim() == 1: audio = audio.unsqueeze(0)
        audio = audio.to(self.mel_basis.device)
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        keyshift_key = str(keyshift) + "_" + str(audio.device)
        if keyshift_key not in self.hann_window: self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(audio.device)
        fft = torch.stft(
            audio, n_fft=n_fft_new, hop_length=hop_length_new, win_length=win_length_new,
            window=self.hann_window[keyshift_key], center=center, return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size: magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        self.mel_basis = self.mel_basis.to(magnitude.device)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec

# --- RMVPEONNXPredictor ---
class RMVPEONNXPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device_str = device
        self.torch_device = torch.device(self._get_torch_device(device))
        logger.info(f"Loading RMVPE ONNX model from {model_path} for device {device}")
        providers = self._get_onnx_providers(device)
        try:
            self.model = ort.InferenceSession(model_path, providers=providers)
            logger.info(f"RMVPE ONNX model loaded with providers: {self.model.get_providers()}")
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name
            logger.debug(f"RMVPE ONNX expects input '{self.input_name}' and provides output '{self.output_name}'")
        except Exception as e:
            logger.error(f"Failed to load RMVPE ONNX model: {e}")
            raise RuntimeError(f"Failed to load RMVPE ONNX model: {e}") from e
        self.mel_extractor = MelSpectrogram(
            n_mel_channels=128, sampling_rate=16000, win_length=1024,
            hop_length=160, n_fft=1024, mel_fmin=30, mel_fmax=8000,
        ).to(self.torch_device)
        self.mel_extractor.eval()
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4)).astype(np.float32)

    def _get_onnx_providers(self, device):
        if device == "cpu": return ["CPUExecutionProvider"]
        elif device.startswith("cuda"): return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "dml": return ["DmlExecutionProvider", "CPUExecutionProvider"]
        else: logger.warning(f"Unsupported device '{device}' for ONNX, falling back to CPU."); return ["CPUExecutionProvider"]

    def _get_torch_device(self, device_str):
         if device_str.startswith("cuda"): return device_str
         elif device_str == "dml": logger.warning("DirectML selected for ONNX, but PyTorch ops (MelSpectrogram) will run on CPU."); return "cpu"
         else: return "cpu"

    def decode(self, hidden, thred=0.03):
        if hidden.ndim == 3: hidden = hidden.squeeze(0)
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[cents_pred < 1e-5] = 0
        return f0.astype(np.float32)

    def to_local_average_cents(self, salience, thred=0.05):
        if not isinstance(salience, np.ndarray): salience = salience.cpu().numpy()
        if salience.ndim == 3: salience = salience.squeeze(0)
        salience = salience.astype(np.float32)
        center_indices = np.argmax(salience, axis=1)
        max_salience_original = np.max(salience, axis=1)
        salience_padded = np.pad(salience, ((0, 0), (4, 4)), mode='constant', constant_values=0)
        center_padded_indices = center_indices + 4
        starts = center_padded_indices - 4
        ends = center_padded_indices + 5
        t_indices = np.arange(salience.shape[0])[:, None]
        window_indices = starts[:, None] + np.arange(9)
        window_indices_clipped = np.clip(window_indices, 0, salience_padded.shape[1] - 1)
        salience_windows = salience_padded[t_indices, window_indices_clipped]
        cents_windows = self.cents_mapping[window_indices_clipped]
        product_sum = np.sum(salience_windows * cents_windows, axis=1)
        weight_sum = np.sum(salience_windows, axis=1)
        epsilon = 1e-8
        weight_sum_safe = np.where(weight_sum < epsilon, epsilon, weight_sum)
        cents_avg = product_sum / weight_sum_safe
        zero_indices = max_salience_original < thred
        cents_avg[zero_indices] = 0.0
        # num_zeroed = np.sum(zero_indices) # Убрано - это debug лог
        # logger.debug(f"RMVPE F0 thresholding: {num_zeroed}/{len(cents_avg)} frames set to 0 (threshold={thred:.3f})")
        return cents_avg.astype(np.float32)

    def compute_f0(self, wav, p_len=None, orig_sr=None):
        if orig_sr is None: raise ValueError("Original sampling rate (orig_sr) must be provided")
        # logger.debug(f"RMVPE compute_f0 started. Input wav shape: {wav.shape}, Original SR: {orig_sr}")
        with torch.no_grad():
            if isinstance(wav, torch.Tensor): wav = wav.cpu().numpy()
            if orig_sr != 16000: wav_16k = librosa.resample(wav, orig_sr=orig_sr, target_sr=16000, res_type='soxr_vhq')
            else: wav_16k = wav
            audio_torch = torch.from_numpy(wav_16k).float().to(self.torch_device)
            if audio_torch.dim() == 1: audio_torch = audio_torch.unsqueeze(0)
            mel = self.mel_extractor(audio_torch, center=True)
            mel = mel.squeeze(0)
            # logger.debug(f"Mel spectrogram computed. Shape: {mel.shape}, Device: {mel.device}, Dtype: {mel.dtype}")
        mel_np = mel.cpu().numpy().astype(np.float32)
        # logger.debug(f"Mel numpy stats: min={np.min(mel_np):.4f}, max={np.max(mel_np):.4f}, mean={np.mean(mel_np):.4f}")
        n_frames = mel_np.shape[1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0: mel_padded = np.pad(mel_np, ((0, 0), (0, n_pad)), mode='constant', constant_values=0)
        else: mel_padded = mel_np
        # logger.debug(f"Padded Mel shape for ONNX: {mel_padded.shape}")
        mel_onnx_input = np.expand_dims(mel_padded, axis=0)
        try:
            # logger.debug("Running RMVPE ONNX inference...")
            hidden = self.model.run([self.output_name], {self.input_name: mel_onnx_input})[0]
            # logger.debug(f"RMVPE ONNX raw output shape: {hidden.shape}, Dtype: {hidden.dtype}")
        except Exception as e:
             logger.error(f"RMVPE ONNX runtime error: {e}", exc_info=True)
             raise RuntimeError(f"RMVPE ONNX runtime error: {e}") from e
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != 360:
             logger.error(f"Unexpected RMVPE ONNX output shape: {hidden.shape}. Expected [1, T_padded, 360]")
             raise ValueError(f"Unexpected RMVPE ONNX output shape: {hidden.shape}")
        hidden = hidden[0, :n_frames, :]
        # logger.debug(f"RMVPE hidden (salience map) shape after clipping: {hidden.shape}")
        # logger.debug(f"Hidden stats: min={np.min(hidden):.4f}, max={np.max(hidden):.4f}, mean={np.mean(hidden):.4f}")
        # logger.debug("Decoding F0 from salience map...")
        f0 = self.decode(hidden, thred=0.03)
        # logger.debug(f"Decoded F0 shape: {f0.shape}")
        # logger.debug(f"Decoded F0 stats: min={np.min(f0):.2f}, max={np.max(f0):.2f}, mean={np.mean(f0):.2f}, non-zero={(f0 > 0).sum()}/{len(f0)}")
        if p_len is not None:
            target_len = p_len
            current_len = len(f0)
            if current_len != target_len:
                # logger.debug(f"Adjusting F0 length from {current_len} to {target_len}")
                x_old = np.linspace(0, 1, current_len)
                x_new = np.linspace(0, 1, target_len)
                f0 = np.interp(x_new, x_old, f0)
        # logger.debug(f"Final F0 shape: {f0.shape}")
        return f0.astype(np.float32)


class ContentVec:
    def __init__(self, vec_path="vec-768-layer-12.onnx", device=None):
        logger.info(f"Loading ContentVec ONNX model from {vec_path}")
        providers = self._get_onnx_providers(device)
        try:
            self.model = ort.InferenceSession(vec_path, providers=providers)
            logger.info(f"ContentVec ONNX model loaded with providers: {self.model.get_providers()}")
        except Exception as e:
            logger.error(f"Failed to load ContentVec ONNX model: {e}")
            raise RuntimeError(f"Failed to load ContentVec ONNX model: {e}") from e

    def _get_onnx_providers(self, device):
        if device == "cpu" or device is None: return ["CPUExecutionProvider"]
        elif device.startswith("cuda"): return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "dml": return ["DmlExecutionProvider", "CPUExecutionProvider"]
        else: logger.warning(f"Unsupported device '{device}' for ONNX, falling back to CPU."); return ["CPUExecutionProvider"]

    def __call__(self, wav):
        return self.forward(wav)

    def forward(self, wav):
        if isinstance(wav, torch.Tensor): wav = wav.cpu().numpy()
        feats = wav.astype(np.float32)
        if feats.ndim == 2: feats = feats.mean(-1)
        assert feats.ndim == 1, feats.ndim
        feats = np.expand_dims(np.expand_dims(feats, 0), 0)
        onnx_input = {self.model.get_inputs()[0].name: feats}
        logits = self.model.run(None, onnx_input)[0]
        return logits.transpose(0, 2, 1)

# --- get_f0_predictor ---
def get_f0_predictor(f0_predictor_str, hop_length, sampling_rate, device, **kargs):
    logger.info(f"Initializing F0 predictor: {f0_predictor_str} with hop_length={hop_length}")
    if f0_predictor_str == "pm":
        from tts_with_rvc.lib.infer_pack.modules.F0Predictor.PMF0Predictor import PMF0Predictor
        f0_predictor_object = PMF0Predictor(hop_length=hop_length, sampling_rate=sampling_rate)
    elif f0_predictor_str == "harvest":
        from tts_with_rvc.lib.infer_pack.modules.F0Predictor.HarvestF0Predictor import HarvestF0Predictor
        f0_predictor_object = HarvestF0Predictor(hop_length=hop_length, sampling_rate=sampling_rate)
    elif f0_predictor_str == "dio":
        from tts_with_rvc.lib.infer_pack.modules.F0Predictor.DioF0Predictor import DioF0Predictor
        f0_predictor_object = DioF0Predictor(hop_length=hop_length, sampling_rate=sampling_rate)
    # elif f0_predictor_str == "crepe":
    #     try:
    #         f0_predictor_object = CrepeF0Predictor(
    #             hop_length=hop_length, sampling_rate=sampling_rate, device=device, threshold=kargs.get('cr_threshold', 0.05)
    #         )
    #     except ImportError:
    #          logger.error("torchcrepe not installed, cannot use 'crepe' F0 predictor. Please install it (`pip install torchcrepe`) or choose another method.")
    #          raise ImportError("torchcrepe not installed, required for 'crepe' F0 predictor.")
    elif f0_predictor_str == "rmvpe":
        rmvpe_model_name = "rmvpe.onnx"
        rmvpe_local_path = os.path.join(os.getcwd(), rmvpe_model_name)
        if not os.path.exists(rmvpe_local_path):
            logger.warning(f"RMVPE ONNX model '{rmvpe_local_path}' not found locally.")
            logger.info(f"Attempting to download '{rmvpe_model_name}' from 'lj1995/VoiceConversionWebUI'...")
            try:
                hf_hub_download(
                    repo_id="lj1995/VoiceConversionWebUI", filename=rmvpe_model_name,
                    local_dir=os.getcwd(), local_dir_use_symlinks=False, token=False
                )
                logger.info(f"Successfully downloaded '{rmvpe_model_name}'.")
                if not os.path.exists(rmvpe_local_path): raise FileNotFoundError("Download success, but file not found.")
            except Exception as e:
                logger.error(f"Failed to download '{rmvpe_model_name}': {e}")
                raise RuntimeError(f"RMVPE ONNX model not found and download failed.") from e
        f0_predictor_object = RMVPEONNXPredictor(rmvpe_local_path, device=device)
    else:
        raise Exception(f"Unknown f0 predictor: {f0_predictor_str}")
    return f0_predictor_object

# --- OnnxRVC ---
class OnnxRVC:
    def __init__(
        self, model_path, sr=40000, hop_size=512,
        vec_path="vec-768-layer-12.onnx", device="cpu", x_pad=3,
    ):
        
        self.device = device
        self.current_rvc_model_path = None
        self.model = None

        logger.info(f"Initializing OnnxRVC: model={model_path}, device={device}, SR={sr}")
        vec_local_path = os.path.join(os.getcwd(), vec_path)
        if not os.path.exists(vec_local_path):
             logger.warning(f"ContentVec model '{vec_path}' not found locally.")
             logger.info(f"Attempting to download '{vec_path}' from 'NaruseMioShirakana/MoeSS-SUBModel'...")
             try:
                 hf_hub_download(repo_id="NaruseMioShirakana/MoeSS-SUBModel", filename=vec_path, local_dir=os.getcwd(), token=False)
                 logger.info(f"Successfully downloaded '{vec_path}'.")
                 if not os.path.exists(vec_local_path): raise FileNotFoundError("Download success, but file not found.")
             except Exception as e:
                 logger.error(f"Failed to download '{vec_path}': {e}")
                 raise RuntimeError(f"ContentVec model not found and download failed.") from e

        self.vec_model = ContentVec(vec_local_path, device)
        
        self.sampling_rate = sr
        self.hop_size = hop_size
        self.device = device
        self.x_pad = x_pad
        self.sr_hubert = 16000
        self.f0_hop_size = int(160 * (self.sampling_rate / self.sr_hubert))
        logger.info(f"Using F0 hop size: {self.f0_hop_size} samples")
        self.pad_seconds = self.x_pad
        self.t_pad_main_sr = int(self.sampling_rate * self.pad_seconds)
        self.t_pad_hubert_sr = int(self.sr_hubert * self.pad_seconds)
        logger.info(f"Padding: {self.pad_seconds}s ({self.t_pad_main_sr} samples @ main SR, {self.t_pad_hubert_sr} samples @ 16k)")

        self.load_new_rvc_model(model_path)

    def _get_onnx_providers(self, device):
        if device == "cpu" or device is None: return ["CPUExecutionProvider"]
        elif device.startswith("cuda"): return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "dml": return ["DmlExecutionProvider", "CPUExecutionProvider"]
        else: logger.warning(f"Unsupported device '{device}', falling back to CPU."); return ["CPUExecutionProvider"]

    def _load_rvc_session(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RVC model file not found: {model_path}")
        providers = self._get_onnx_providers(self.device)
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            self.model = ort.InferenceSession(model_path, providers=providers)
            self.current_rvc_model_path = model_path
        except Exception as e:
            self.model = None
            self.current_rvc_model_path = None
            raise RuntimeError(f"Failed to load RVC ONNX session: {e}") from e
        
    def load_new_rvc_model(self, model_path):
        if self.current_rvc_model_path == model_path and self.model is not None:
            logger.info(f"Model '{model_path}' is already loaded. Skipping reload.")
            return
        self._load_rvc_session(model_path)

    def load_index(self, index_path):
        if not index_path or not os.path.exists(index_path):
            logger.info(f"Index file not found or not provided: {index_path}. Skipping index.")
            return None, None
        try:
            logger.info(f"Loading Faiss index from {index_path}")
            index = faiss.read_index(index_path)
            if index.ntotal == 0: logger.warning(f"Faiss index '{index_path}' is empty."); return None, None
            big_npy = index.reconstruct_n(0, index.ntotal)
            logger.info(f"Faiss index loaded: {index.ntotal} vectors, dim={index.d}.")
            return index, big_npy.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to load Faiss index '{index_path}': {e}")
            return None, None

    def apply_index(self, hubert_features, index, big_npy, index_rate):
        if index is None or big_npy is None or index_rate == 0: return hubert_features.astype(np.float32)
        try:
            hubert_dim = hubert_features.shape[1]
            index_dim = index.d
            if hubert_dim != index_dim:
                logger.error(f"Dimension mismatch: Hubert({hubert_dim}) != Index({index_dim}). Skipping index.")
                return hubert_features.astype(np.float32)
            hubert_np = hubert_features.squeeze(0).transpose(1, 0).astype(np.float32)
            k = 8
            # logger.debug(f"Searching Faiss index (dim: {index_dim}) with {hubert_np.shape[0]} vectors...")
            D, I = index.search(hubert_np, k=k)
            # logger.debug("Faiss search completed.")
            retrieved_vectors = big_npy[I].astype(np.float32)
            epsilon = 1e-8
            squared_distances = D.astype(np.float32)**2 + epsilon
            weights = 1.0 / squared_distances
            weights_sum = np.sum(weights, axis=1, keepdims=True)
            weights_normalized = weights / np.where(weights_sum == 0, epsilon, weights_sum)
            weighted_sum_vectors = np.sum(retrieved_vectors * weights_normalized[:, :, None], axis=1)
            mixed_hubert = (1 - index_rate) * hubert_np + index_rate * weighted_sum_vectors
            # logger.debug("Faiss index applied successfully.")
            return np.expand_dims(mixed_hubert.transpose(1, 0), axis=0).astype(np.float32)
        except Exception as e:
            logger.error(f"Error applying Faiss index: {e}", exc_info=True)
            return hubert_features.astype(np.float32)

    def apply_protection(self, hubert_features, pitchf, protect_rate):
        if protect_rate >= 0.5 or pitchf is None: return hubert_features.astype(np.float32)
        if isinstance(pitchf, torch.Tensor): pitchf = pitchf.cpu().numpy()
        if pitchf.ndim == 1: pitchf = np.expand_dims(pitchf, axis=0)
        if isinstance(hubert_features, torch.Tensor): hubert_features = hubert_features.cpu().numpy()
        protect_mask = np.where(pitchf > 0, 1.0, protect_rate).astype(np.float32)
        protect_mask = np.expand_dims(protect_mask, axis=1)
        if hubert_features.shape[2] != pitchf.shape[1]:
            # logger.warning(f"Hubert length ({hubert_features.shape[2]}) differs from pitchf length ({pitchf.shape[1]}). Interpolating Hubert for protection.")
            hubert_torch = torch.from_numpy(hubert_features)
            hubert_interp_torch = F.interpolate(hubert_torch.cpu(), size=pitchf.shape[1], mode='linear', align_corners=False)
            hubert_interp_np = hubert_interp_torch.numpy()
            protected_hubert = hubert_interp_np * protect_mask
        else:
             protected_hubert = hubert_features * protect_mask
        return protected_hubert.astype(np.float32)

    def forward(self, hubert, hubert_length, pitch, pitchf, ds, rnd):
        hubert_transposed = hubert.transpose(0, 2, 1)
        onnx_input = {
            self.model.get_inputs()[0].name: hubert_transposed.astype(np.float32),
            self.model.get_inputs()[1].name: hubert_length.astype(np.int64),
            self.model.get_inputs()[2].name: pitch.astype(np.int64),
            self.model.get_inputs()[3].name: pitchf.astype(np.float32),
            self.model.get_inputs()[4].name: ds.astype(np.int64),
            self.model.get_inputs()[5].name: rnd.astype(np.float32),
        }
        try:
            output_wav = self.model.run(None, onnx_input)[0]
            return output_wav
        except Exception as e:
             logger.error(f"ONNX runtime error during forward pass: {e}", exc_info=True)
             raise RuntimeError(f"ONNX runtime error: {e}") from e

    def inference(
        self, raw_path, sid, f0_method="dio", f0_up_key=0, index_file=None,
        index_file2=None, index_rate=0.75, filter_radius=3, resample_sr=0,
        rms_mix_rate=0.25, protect=0.33, cr_threshold=0.05, verbose=False
    ):
        if verbose: logger.parent.setLevel(logging.DEBUG) # Устанавливаем DEBUG уровень для корневого логгера, если verbose=True
        else: logger.parent.setLevel(logging.INFO) # Иначе ставим INFO

        logger.info(f"Starting ONNX inference for: {raw_path}")
        try:
            wav_orig, sr_orig = librosa.load(raw_path, sr=None, mono=True)
            logger.info(f"Loaded audio: SR={sr_orig}, Len={len(wav_orig)} samples")
            if sr_orig != self.sampling_rate:
                logger.info(f"Resampling audio from {sr_orig} to {self.sampling_rate}")
                wav_main_sr = librosa.resample(wav_orig, orig_sr=sr_orig, target_sr=self.sampling_rate, res_type='soxr_vhq')
            else:
                wav_main_sr = wav_orig
        except Exception as e:
             logger.error(f"Failed to load/resample audio: {e}")
             raise RuntimeError(f"Failed to load/resample audio: {e}") from e

        original_length = len(wav_main_sr)
        logger.info(f"Padding audio with {self.t_pad_main_sr} samples")
        wav_padded_main = np.pad(wav_main_sr, (self.t_pad_main_sr, self.t_pad_main_sr), mode='reflect')
        wav16k_orig = librosa.resample(wav_main_sr, orig_sr=self.sampling_rate, target_sr=self.sr_hubert, res_type='soxr_vhq')
        wav16k_padded = np.pad(wav16k_orig, (self.t_pad_hubert_sr, self.t_pad_hubert_sr), mode='reflect')

        pitch_length_padded = wav_padded_main.shape[0] // self.f0_hop_size
        f0_predictor = get_f0_predictor(
            f0_method, hop_length=self.f0_hop_size, sampling_rate=self.sampling_rate,
            device=self.device, cr_threshold=cr_threshold,
        )
        logger.info(f"Computing F0 using {f0_method} (target_len={pitch_length_padded})...")
        compute_f0_kwargs = {"wav": wav_padded_main, "p_len": pitch_length_padded}
        if isinstance(f0_predictor, RMVPEONNXPredictor): compute_f0_kwargs["orig_sr"] = self.sampling_rate
        try:
            pitchf = f0_predictor.compute_f0(**compute_f0_kwargs)
            logger.debug(f"Computed F0 length: {len(pitchf)}")
        except Exception as e:
             logger.error(f"Error during F0 computation with {f0_method}: {e}", exc_info=True)
             raise RuntimeError(f"F0 computation failed: {e}") from e

        if filter_radius >= 3:
            logger.debug(f"Applying median filter(radius={filter_radius}) to F0")
            f0_padded_filter = np.pad(pitchf, int((filter_radius - 1) / 2), mode='reflect')
            pitchf = signal.medfilt(f0_padded_filter, filter_radius)[int((filter_radius - 1) / 2) : -int((filter_radius - 1) / 2)]

        pitchf = pitchf * (2 ** (f0_up_key / 12))
        pitch = pitchf.copy()
        f0_min = 50; f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700); f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0_mel = 1127 * np.log(1 + pitch / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1; f0_mel[f0_mel > 255] = 255
        pitch = np.rint(f0_mel).astype(np.int64)

        logger.debug("Extracting Hubert features...")
        hubert_features = self.vec_model(wav16k_padded)
        hubert_features = np.repeat(hubert_features, 2, axis=2)
        hubert_length = hubert_features.shape[2]
        logger.debug(f"Hubert features extracted. Shape: {hubert_features.shape}")

        if hubert_length != len(pitch):
            if verbose:
                logger.warning(f"Hubert length ({hubert_length}) != Pitch length ({len(pitch)}). Adjusting...")
            target_len = hubert_length
            if len(pitch) < target_len: pitch = np.pad(pitch, (0, target_len - len(pitch)), mode='constant', constant_values=pitch[-1] if len(pitch) > 0 else 1)
            elif len(pitch) > target_len: pitch = pitch[:target_len]
            if len(pitchf) < target_len: x_old = np.linspace(0, 1, len(pitchf)); x_new = np.linspace(0, 1, target_len); pitchf = np.interp(x_new, x_old, pitchf)
            elif len(pitchf) > target_len: pitchf = pitchf[:target_len]
            logger.debug(f"Adjusted Pitch/Pitchf length to: {target_len}")

        current_index_path = index_file
        if not current_index_path and index_file2: logger.info(f"Using secondary index file: {index_file2}"); current_index_path = index_file2
        index, big_npy = self.load_index(current_index_path)
        if index is not None and index_rate > 0: logger.debug(f"Applying Faiss index (rate={index_rate})..."); hubert_features = self.apply_index(hubert_features, index, big_npy, index_rate)
        else: logger.debug("Skipping Faiss index.")

        logger.debug(f"Applying protection (rate={protect})...")
        hubert_features = self.apply_protection(hubert_features, pitchf, protect)

        pitch = pitch.reshape(1, hubert_length)
        pitchf_final = pitchf.reshape(1, hubert_length).astype(np.float32)
        ds = np.array([sid]).astype(np.int64)
        rnd = np.random.randn(1, 192, hubert_length).astype(np.float32)
        hubert_length_np = np.array([hubert_length]).astype(np.int64)

        logger.info("Running main RVC ONNX inference...")
        out_wav_padded = self.forward(hubert_features, hubert_length_np, pitch, pitchf_final, ds, rnd).squeeze()
        logger.debug(f"ONNX inference completed. Output shape (padded): {out_wav_padded.shape}")

        logger.debug(f"Trimming padding ({self.t_pad_main_sr} samples) from output.")
        trim_start = self.t_pad_main_sr
        trim_end = trim_start + original_length
        trim_end = min(trim_end, len(out_wav_padded))
        out_wav = out_wav_padded[trim_start : trim_end]
        if len(out_wav) < original_length: logger.warning(f"Output length {len(out_wav)} < original {original_length}. Padding..."); out_wav = np.pad(out_wav, (0, original_length - len(out_wav)), 'constant')
        elif len(out_wav) > original_length: logger.warning(f"Output length {len(out_wav)} > original {original_length}. Clipping..."); out_wav = out_wav[:original_length]
        logger.debug(f"Output shape after trimming: {out_wav.shape}")

        if rms_mix_rate < 1.0: logger.debug(f"Applying RMS mixing (rate={rms_mix_rate})..."); out_wav = change_rms(wav_main_sr, self.sampling_rate, out_wav, self.sampling_rate, rms_mix_rate)
        final_sr = self.sampling_rate
        if resample_sr > 0 and resample_sr != self.sampling_rate: logger.info(f"Resampling output to {resample_sr} Hz"); out_wav = librosa.resample(out_wav, orig_sr=self.sampling_rate, target_sr=resample_sr, res_type='soxr_vhq'); final_sr = resample_sr
        audio_max = np.abs(out_wav).max() / 0.99
        if audio_max > 1: out_wav /= audio_max
        out_wav = (out_wav * 32767).astype(np.int16)

        logger.info(f"Inference completed. Final audio length: {len(out_wav)} samples, SR: {final_sr}")

        if verbose: logger.parent.setLevel(logging.INFO)
        
        return out_wav

# --- Добавление CrepeF0Predictor (закомментировано) ---
# Нужно раскомментировать и убедиться, что torchcrepe установлен, если нужен этот метод
# try:
#     import torchcrepe
#     _crepe_available = True
# except ImportError:
#     _crepe_available = False
#     logger.info("torchcrepe not found. 'crepe' F0 predictor will not be available.")

# if _crepe_available:
#     class CrepeF0Predictor:
#         # ... (код CrepeF0Predictor) ...
#         pass
# else:
#     # Заглушка, если crepe недоступен
#     class CrepeF0Predictor:
#         def __init__(self, *args, **kwargs):
#             raise NotImplementedError("CrepeF0Predictor requires torchcrepe to be installed.")
#         def compute_f0(self, *args, **kwargs):
#              raise NotImplementedError("CrepeF0Predictor requires torchcrepe to be installed.")