import traceback
import logging
import sys
import locale
import re

# import os

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from io import BytesIO
import os

from infer.lib.audio import load_audio, wav2
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import *

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

class VC:

    def clean_filename(self, filename):
        return re.sub(r"[^\w\s-]", "", filename)
    
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

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
            ),
            "__type__": "update",
        }

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
            return (
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": to_return_protect1,
                    "__type__": "update",
                },
                "",
                "",
            )
        person = f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading: {person}")

        self.cpt = torch.load(person, map_location="cpu")
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
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        logger.info("Select index: " + index["value"])

        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )

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
        spk_item,
        output_format="wav",
        opt_root="",  # 传入保存路径
    ):
        if input_audio_path is None:
            return "You need to upload an audio", None

        f0_up_key = int(f0_up_key)

        try:
            # 加载音频
            audio = load_audio(input_audio_path, 16000)

            # 规范化音频
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            # 加载模型
            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)

            # 处理 file_index
            if file_index:
                file_index = file_index.strip().replace("trained", "added")
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""

            # 确保管道已经初始化
            if self.pipeline is None:
                return "Pipeline not initialized", None

            # 调用 pipeline
            try:
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
                    # spk_item,  # 如果这个参数需要传递，请确保它在 `pipeline` 方法中定义
                )
            except TypeError as e:
                return f"Pipeline error: {str(e)}", None

            # 确定目标采样率
            tgt_sr = resample_sr if self.tgt_sr != resample_sr >= 16000 else self.tgt_sr

            # 处理索引信息
            index_info = (
                f"Index:\n{file_index}."
                if os.path.exists(file_index)
                else "Index not used."
            )

            # 使用 opt_root 目录
            if opt_root:
                os.makedirs(opt_root, exist_ok=True)
            else:
                opt_root = "results"
                if not os.path.exists(opt_root):
                    os.makedirs(opt_root)

            # 生成输出文件名
            truncated_basename = self.clean_filename(Path(input_audio_path).stem)
            spk_item_name = self.clean_filename(os.path.splitext(spk_item)[0])
            output_file_name = f"{truncated_basename}_{spk_item_name}_{f0_method}_{f0_up_key}key.{output_format}"
            output_file_path = os.path.join(opt_root, output_file_name)  # 使用 opt_root 目录

            # 检查文件是否已存在，添加计数后缀
            count = 1
            while os.path.exists(output_file_path):
                output_file_name = f"{truncated_basename}_{spk_item_name}_{f0_method}_{f0_up_key}key_{count}.{output_format}"
                output_file_path = os.path.join(opt_root, output_file_name)
                count += 1

            # 写入音频文件
            sf.write(output_file_path, audio_opt, tgt_sr, format=output_format)

            return (
                f"Success.\n{index_info}\nTime:\nnpy: {times[0]:.2f}s, f0: {times[1]:.2f}s, infer: {times[2]:.2f}s.",
                output_file_path,
            )
        except Exception as e:
            info = traceback.format_exc()
            logger.warning(info)
            return str(e), None



    def vc_multi(
            self,
            sid,
            dir_path,
            opt_root,
            paths,
            f0_up_key,
            f0_method,
            file_index,
            file_index2,
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect,
            format1,
            spk_item
        ):
        try:
            # 处理路径
            dir_path = dir_path.strip().strip('"').strip("\n").strip()
            opt_root = opt_root.strip().strip('"').strip("\n").strip()
            os.makedirs(opt_root, exist_ok=True)

            # 获取文件路径列表
            try:
                if dir_path:
                    paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
                else:
                    paths = [path.name for path in paths]
            except:
                traceback.print_exc()
                paths = [path.name for path in paths]

            infos = []
            for path in paths:
                # 调用 vc_single 函数
                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    file_index,
                    file_index2,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                    spk_item,
                    format1,  # 确保传递了所有必要参数
                    opt_root,
                )

                if "Success" in info:
                    try:
                        # 处理路径和文件名
                        truncated_basename = self.clean_filename(Path(path).stem)
                        spk_item_name = self.clean_filename(os.path.splitext(spk_item)[0])
                        output_file_name = f"{truncated_basename}_{spk_item_name}_{f0_method}_{f0_up_key}key.{format1}"
                        output_file_path = os.path.join(opt_root, output_file_name)
                        
                        # 如果文件名已存在，添加计数后缀
                        count = 1
                        while os.path.exists(output_file_path):
                            output_file_name = f"{truncated_basename}_{spk_item_name}_{f0_method}_{f0_up_key}key_{count}.{format1}"
                            output_file_path = os.path.join(opt_root, output_file_name)
                            count += 1

                        # 确保 opt 是一个包含两个元素的元组
                        if isinstance(opt, tuple) and len(opt) == 2:
                            tgt_sr, audio_opt = opt
                            if format1 in ["wav", "flac"]:
                                sf.write(output_file_path, audio_opt, tgt_sr, format=format1)
                            else:
                                with BytesIO() as wavf:
                                    sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                    wavf.seek(0, 0)
                                    with open(output_file_path, "wb") as outf:
                                        wav2(wavf, outf, format1)
                        else:
                            info += "Unexpected output format from vc_single."

                    except Exception as e:
                        info += traceback.format_exc()

                infos.append(f"{os.path.basename(path)} -> {info}")

            yield "\n".join(infos)
        except Exception as e:
            yield traceback.format_exc()


