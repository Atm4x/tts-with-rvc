import traceback
import logging

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import torch
from io import BytesIO
import os

from huggingface_hub import hf_hub_download
from tts_with_rvc.infer.lib.audio import load_audio, wav2
from tts_with_rvc.infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from tts_with_rvc.infer.vc.pipeline import Pipeline

from fairseq import checkpoint_utils


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None

        self.config = config

    def get_vc(self, sid):
        logger.info("Get sid: " + sid)

        if sid == "" or sid == []:
            if (
                self.hubert_model is not None
            ):  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
                logger.info("Clean model cache")
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)  # ,cpt
                self.hubert_model = self.net_g = self.n_spk = self.hubert_model = (
                    self.tgt_sr
                ) = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ###楼下不这么折腾清理不干净
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return
        logger.info(f"Loading: {sid}")

        self.cpt = torch.load(sid, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)

    def load_hubert(self, config, file_path="hubert_base.pt"):
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [file_path],
            suffix="",
        )
        hubert_model = models[0]
        hubert_model = hubert_model.to(config.device)
        print(config.device)
        if config.is_half:
            hubert_model = hubert_model.half()
        else:
            hubert_model = hubert_model.float()
        return hubert_model.eval()

    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        if input_audio_path is None:
            return "You need to upload an audio", None
        f0_up_key = int(f0_up_key)
        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                hubert_name = "hubert_base.pt"
                if not os.path.exists(os.path.join(os.getcwd(), hubert_name)):
                    hf_hub_download(repo_id="lj1995/VoiceConversionWebUI", filename=hubert_name, local_dir=os.getcwd(), token=False)
                self.hubert_model = self.load_hubert(self.config, hubert_name)

            if file_index:
                file_index = (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""  # 防止小白写错，自动帮他替换掉

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                f0_file,
            )
            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            index_info = (
                "Index:\n%s." % file_index
                if os.path.exists(file_index)
                else "Index not used."
            )
            # print("Success.\n%s\nTime:\nnpy: %.2fs, f0: %.2fs, infer: %.2fs." % (index_info, *times))
            return tgt_sr, audio_opt
        except:
            info = traceback.format_exc()
            logger.warning(info)
            return info, (None, None)

    def vc_stream(
        self,
        sid,
        audio,  # This accepts audio data directly, not a path
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        """
        Voice conversion for streaming audio data.
        Similar to vc_single but takes audio data directly instead of loading from a path.
        """
        f0_up_key = int(f0_up_key)
        try:
            # No need to load audio from disk, it's already provided
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                hubert_name = "hubert_base.pt"
                if not os.path.exists(os.path.join(os.getcwd(), hubert_name)):
                    hf_hub_download(repo_id="lj1995/VoiceConversionWebUI", filename=hubert_name, local_dir=os.getcwd(), token=False)
                self.hubert_model = self.load_hubert(self.config, hubert_name)

            if file_index:
                file_index = (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""

            # For streaming, we pass None as input_audio_path since we don't have a file path
            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                None,  # input_audio_path is None for streaming
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                f0_file,
            )
            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            index_info = (
                "Index:\n%s." % file_index
                if os.path.exists(file_index)
                else "Index not used."
            )
            return tgt_sr, audio_opt
        except:
            info = traceback.format_exc()
            logger.warning(info)
            return info, (None, None)
    
