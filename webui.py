"""LongCat-AudioDiT Gradio WebUI — Full-featured TTS with voice cloning, SSML, and advanced controls."""

import re
import io
import sys
import json
import wave
import datetime
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import librosa
import numpy as np
import torch
import torchaudio
import soundfile as sf

# Register audiodit
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audiodit  # noqa: F401
from audiodit import AudioDiTModel
from transformers import AutoTokenizer

torch.backends.cudnn.benchmark = False

# ─── SenseVoice ASR (lazy load) ────────────────────────────
_asr_model = None


def get_asr_model():
    global _asr_model
    if _asr_model is None:
        from funasr import AutoModel
        print("Loading SenseVoice ASR model ...")
        _asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_kwargs={"max_single_segment_time": 30000},
            trust_remote_code=True,
        )
        print("SenseVoice ASR model loaded.")
    return _asr_model


# ASR emoji processing (from official SenseVoice webui)
_emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "",
    "<|zh|>": "", "<|en|>": "", "<|yue|>": "", "<|ja|>": "", "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "", "<|SAD|>": "", "<|ANGRY|>": "", "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "", "<|DISGUSTED|>": "", "<|SURPRISED|>": "",
    "<|BGM|>": "", "<|Speech|>": "", "<|Applause|>": "", "<|Laughter|>": "",
    "<|Cry|>": "", "<|Sneeze|>": "", "<|Breath|>": "", "<|Cough|>": "",
    "<|Sing|>": "", "<|Speech_Noise|>": "", "<|withitn|>": "", "<|woitn|>": "",
    "<|GBG|>": "", "<|Event_UNK|>": "", "<|EMO_UNKNOWN|>": "",
}

_emo_dict = {"<|HAPPY|>": "", "<|SAD|>": "", "<|ANGRY|>": "", "<|NEUTRAL|>": "",
             "<|FEARFUL|>": "", "<|DISGUSTED|>": "", "<|SURPRISED|>": ""}
_event_dict = {"<|BGM|>": "", "<|Speech|>": "", "<|Applause|>": "", "<|Laughter|>": "",
               "<|Cry|>": "", "<|Sneeze|>": "", "<|Breath|>": "", "<|Cough|>": ""}
_emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
_event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷"}
_lang_dict = {"<|zh|>": "<|lang|>", "<|en|>": "<|lang|>", "<|yue|>": "<|lang|>",
              "<|ja|>": "<|lang|>", "<|ko|>": "<|lang|>", "<|nospeech|>": "<|lang|>"}


def _format_str_v2(s):
    sptk_dict = {}
    for sptk in _emoji_dict:
        sptk_dict[sptk] = s.count(sptk)
        s = s.replace(sptk, _emoji_dict[sptk])
    emo = "<|NEUTRAL|>"
    for e in _emo_dict:
        if sptk_dict.get(e, 0) > sptk_dict.get(emo, 0):
            emo = e
    for e in _event_dict:
        if sptk_dict.get(e, 0) > 0:
            s = _event_dict[e] + s
    s = s + _emo_dict.get(emo, "")
    for emoji in _emo_set.union(_event_set):
        s = s.replace(" " + emoji, emoji)
        s = s.replace(emoji + " ", emoji)
    return s.strip()


def _clean_asr_text(s):
    """清理 ASR 输出的特殊标记，返回纯文本"""
    s = s.replace("<|nospeech|><|Event_UNK|>", "")
    for lang in _lang_dict:
        s = s.replace(lang, "<|lang|>")
    s_list = [_format_str_v2(si).strip() for si in s.split("<|lang|>")]
    new_s = " ".join(sl for sl in s_list if sl)
    # 去掉 emoji
    new_s = re.sub(r'[😊😔😡😰🤢😮🎼👏😀😭🤧😷❓]', '', new_s)
    return new_s.strip()


def transcribe_audio(audio_input) -> str:
    """ASR: 音频转文本"""
    if audio_input is None:
        return ""
    try:
        asr = get_asr_model()
        fs, wav = audio_input
        wav = wav.astype(np.float32)
        if np.abs(wav).max() > 1.0:
            wav = wav / np.iinfo(np.int16).max
        if len(wav.shape) > 1:
            wav = wav.mean(-1)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            wav = resampler(torch.from_numpy(wav).to(torch.float32)[None, :])[0, :].numpy()
        result = asr.generate(input=wav, cache={}, language="auto", use_itn=True, batch_size_s=60, merge_vad=True)
        text = result[0]["text"]
        text = _clean_asr_text(text)
        return text
    except Exception as e:
        print(f"ASR error: {e}")
        return f"[ASR识别失败: {e}]"

MAX_SEED = 2**32 - 1
EN_DUR_PER_CHAR = 0.082
ZH_DUR_PER_CHAR = 0.21

# ─── Model paths ──────────────────────────────────────────────
LOCAL_MODEL_MAP = {
    "3.5B": "/home/lonren/.cache/huggingface/hub/meituan-longcat/LongCat-AudioDiT-3.5B",
}
MODEL_CACHE = {}


# ─── Text normalization ───────────────────────────────────────

def fullwidth_to_halfwidth(text: str) -> str:
    """全角数字/字母转半角"""
    result = []
    for c in text:
        code = ord(c)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            result.append(' ')
        else:
            result.append(c)
    return ''.join(result)


def filter_special_symbols(text: str) -> str:
    """过滤包裹数字的特殊符号"""
    text = re.sub(r'[\[\]()（）【】]', '', text)
    return text


DIGIT_MAP = {
    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
}
TELEPHONE_MAP = {
    '0': '零', '1': '幺', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
}


def number_to_digits_text(s: str) -> str:
    """123 → 一二三"""
    return ''.join(DIGIT_MAP.get(c, c) for c in s)


def number_to_cardinal_text(s: str) -> str:
    """123 → 一百二十三"""
    try:
        n = int(s)
    except ValueError:
        try:
            n = float(s)
        except ValueError:
            return s
    return _int_to_chinese(int(n))


def _int_to_chinese(num: int) -> str:
    if num == 0:
        return '零'
    units = ['', '十', '百', '千', '万']
    digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    if num < 0:
        return '负' + _int_to_chinese(-num)
    if num >= 100000000:
        return _int_to_chinese(num // 100000000) + '亿' + (
            _int_to_chinese(num % 100000000) if num % 100000000 else '')
    result = ''
    zero_flag = False
    for i in range(len(units)):
        if num == 0:
            break
        n = num % 10
        if n == 0:
            zero_flag = True
        else:
            if zero_flag:
                result = digits[0] + result
                zero_flag = False
            result = digits[n] + units[i] + result
        num //= 10
    return result


def number_to_telephone_text(s: str) -> str:
    """138 → 幺三八"""
    return ''.join(TELEPHONE_MAP.get(c, c) for c in s)


def number_to_date_text(s: str) -> str:
    """20260425 → 二零二六年四月二十五日"""
    if len(s) == 8 and s.isdigit():
        y, m, d = s[:4], s[4:6], s[6:8]
        return f"{number_to_digits_text(y)}年{int(m)}月{int(d)}日"
    elif len(s) == 6 and s.isdigit():
        y, m = s[:4], s[4:6]
        return f"{number_to_digits_text(y)}年{int(m)}月"
    return number_to_digits_text(s)


def number_to_currency_text(s: str) -> str:
    """19.9 → 十九点九元"""
    try:
        f = float(s)
    except ValueError:
        return s
    int_part = int(f)
    dec_part = str(s).split('.')[-1] if '.' in s else ''
    result = _int_to_chinese(int_part) if int_part > 0 else '零'
    if dec_part:
        result += '点' + number_to_digits_text(dec_part)
    result += '元'
    return result


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[\u201c\u201d\u201e\u2018\u2019]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_text(text: str, do_fullwidth: bool = True, do_filter_symbols: bool = True) -> str:
    if do_fullwidth:
        text = fullwidth_to_halfwidth(text)
    if do_filter_symbols:
        text = filter_special_symbols(text)
    return text


def segment_text(text: str, max_chars: int = 200) -> list:
    """按标点分段，每段不超过 max_chars 字符

    逻辑：按标点（中英文句号/问号/感叹号/逗号/分号/冒号/换行）切分，
    合并短句直到接近 max_chars 时断开。
    如果单个标点片段超过 max_chars，强制在 max_chars 处断开。
    """
    if len(text) <= max_chars:
        return [text]

    # 按所有标点切分，保留标点
    parts = re.split(r'([。！？，；：、\n\.\?\!\,\;\:])', text)

    # 合并片段
    segments = []
    current = ''
    i = 0
    while i < len(parts):
        part = parts[i]
        # 如果下一个是标点，合并进来
        if i + 1 < len(parts) and re.match(r'[。！？，；：、\n\.\?\!\,\;\:]', parts[i + 1]):
            part += parts[i + 1]
            i += 2
        else:
            i += 1

        if not part.strip():
            continue

        if len(current) + len(part) <= max_chars:
            current += part
        else:
            # 当前段已满，保存
            if current.strip():
                segments.append(current.strip())
            # 新片段本身就超长，强制截断
            if len(part) > max_chars:
                for j in range(0, len(part), max_chars):
                    chunk = part[j:j + max_chars]
                    if chunk.strip():
                        segments.append(chunk.strip())
                current = ''
            else:
                current = part

    if current.strip():
        segments.append(current.strip())

    return segments if segments else [text]


# ─── SSML Parser ──────────────────────────────────────────────

def parse_ssml(ssml: str, number_mode: str = "auto") -> str:
    """解析SSML标签，转换为纯文本

    支持标签:
    - <say-as interpret-as="digits|cardinal|telephone|date|currency">数字</say-as>
    - <speak>...</speak> (根标签，直接去除)
    """
    text = ssml.strip()

    # 去掉 <speak> 根标签
    text = re.sub(r'</?speak[^>]*>', '', text)

    # 处理 <say-as> 标签
    def replace_say_as(match):
        attrs = match.group(1) or ''
        content = match.group(2)
        interpret = 'auto'
        m = re.search(r'interpret-as=["\']([^"\']+)["\']', attrs)
        if m:
            interpret = m.group(1)
        mode = interpret if interpret != 'auto' else number_mode
        return _convert_numbers_in_text(content, mode)

    text = re.sub(r'<say-as([^>]*)>(.*?)</say-as>', replace_say_as, text, flags=re.DOTALL)
    return text


def _convert_numbers_in_text(text: str, mode: str) -> str:
    """根据模式转换文本中的数字"""
    if mode == 'auto':
        return text

    def convert_match(m):
        num_str = m.group(0)
        if mode == 'digits':
            return number_to_digits_text(num_str.replace('.', ''))
        elif mode == 'cardinal':
            return number_to_cardinal_text(num_str)
        elif mode == 'telephone':
            return number_to_telephone_text(num_str)
        elif mode == 'date':
            return number_to_date_text(num_str)
        elif mode == 'currency':
            return number_to_currency_text(num_str)
        return num_str

    # 匹配数字（含小数点）
    return re.sub(r'\d+\.?\d*', convert_match, text)


def auto_convert_numbers(text: str, mode: str) -> str:
    """自动转换文本中的数字（非SSML模式）"""
    if mode == 'auto':
        return text
    return _convert_numbers_in_text(text, mode)


# ─── Audio processing ─────────────────────────────────────────

def adjust_speed(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """调整语速 (0.5-2.0)"""
    if abs(speed - 1.0) < 0.01:
        return audio
    speed_factor = 0.5 + speed * 1.5 / 100  # speed 0-100 → 0.5-2.0
    return librosa.effects.time_stretch(audio, rate=speed_factor)


def adjust_volume(audio: np.ndarray, volume: int) -> np.ndarray:
    """调整音量 (0-100, 50=原始)"""
    if volume == 50:
        return audio
    factor = (volume / 50.0)
    return np.clip(audio * factor, -1.0, 1.0)


def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """去除首尾静音"""
    if len(audio) == 0:
        return audio
    indices = np.where(np.abs(audio) > threshold)[0]
    if len(indices) == 0:
        return audio
    return audio[indices[0]:indices[-1] + 1]


def apply_agc(audio: np.ndarray, target_level: float = 0.9) -> np.ndarray:
    """自动增益控制"""
    peak = np.abs(audio).max()
    if peak > 0:
        gain = target_level / peak
        gain = min(gain, 10.0)  # 防止过度放大
        return audio * gain
    return audio


def audio_to_bytes(audio: np.ndarray, sr: int, fmt: str = 'wav') -> bytes:
    """转换音频为指定格式的字节"""
    buf = io.BytesIO()
    if fmt == 'mp3':
        sf.write(buf, audio, sr, format='MP3')
    else:
        sf.write(buf, audio, sr, format='WAV', subtype='PCM_16')
    return buf.getvalue()


# ─── Duration estimation ──────────────────────────────────────

def approx_duration_from_text(text: str, max_duration: float = 30.0) -> float:
    text = re.sub(r'\s+', '', text)
    num_zh = num_en = num_other = 0
    for c in text:
        if '\u4e00' <= c <= '\u9fff':
            num_zh += 1
        elif c.isalpha():
            num_en += 1
        else:
            num_other += 1
    if num_zh > num_en:
        num_zh += num_other
    else:
        num_en += num_other
    return min(max_duration, num_zh * ZH_DUR_PER_CHAR + num_en * EN_DUR_PER_CHAR)


# ─── Model management ─────────────────────────────────────────

def get_model(model_name: str):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    # 卸载其他已加载模型以释放显存
    for old_name in list(MODEL_CACHE.keys()):
        if old_name != model_name:
            old_model, old_tokenizer = MODEL_CACHE.pop(old_name)
            del old_model, old_tokenizer
            torch.cuda.empty_cache()
            import gc; gc.collect()
            print(f"Unloaded model {old_name} to free VRAM")
    model_id = LOCAL_MODEL_MAP.get(model_name, f"meituan-longcat/LongCat-AudioDiT-{model_name}")
    print(f"Loading model {model_id} ...")
    model = AudioDiTModel.from_pretrained(model_id).to("cuda")
    model.vae.to_half()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    MODEL_CACHE[model_name] = (model, tokenizer)
    print(f"Model {model_id} loaded.")
    return model, tokenizer


# ─── Inference ─────────────────────────────────────────────────

def generate_tts_core(
    text: str,
    model_name: str,
    guidance_method: str,
    nfe: int,
    guidance_strength: float,
    seed: int,
    prompt_audio_wav: Optional[torch.Tensor] = None,
    prompt_text: Optional[str] = None,
    max_duration_sec: float = 30.0,
) -> Tuple[int, np.ndarray]:
    """核心生成函数"""
    model, tokenizer = get_model(model_name)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    text = normalize_text(text)
    sr = model.config.sampling_rate
    full_hop = model.config.latent_hop
    max_duration = model.config.max_wav_duration

    if prompt_audio_wav is not None and prompt_text:
        full_text = f"{normalize_text(prompt_text)} {text}"
    else:
        full_text = text

    inputs = tokenizer([full_text], padding="longest", return_tensors="pt")
    dur_sec = approx_duration_from_text(text, max_duration=max_duration_sec)

    duration = int(dur_sec * sr // full_hop)

    gen_kwargs = dict(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        duration=duration,
        steps=nfe,
        cfg_strength=guidance_strength,
        guidance_method=guidance_method,
    )

    if prompt_audio_wav is not None:
        gen_kwargs["prompt_audio"] = prompt_audio_wav

        # Recalculate duration for voice cloning (matching official inference.py)
        _, prompt_dur = model.encode_prompt_audio(prompt_audio_wav)
        prompt_time = prompt_dur * full_hop / sr
        print(f"[CLONE] prompt_dur={prompt_dur}, prompt_time={prompt_time:.2f}s, prompt_wav shape={prompt_audio_wav.shape}, dtype={prompt_audio_wav.dtype}")
        # 减去 prompt 占用的时间
        dur_sec = approx_duration_from_text(text, max_duration=max_duration_sec - prompt_time)
        if prompt_text:
            approx_pd = approx_duration_from_text(prompt_text, max_duration=max_duration)
            ratio = np.clip(prompt_time / max(approx_pd, 0.1), 1.0, 1.5)
            dur_sec = dur_sec * ratio
        duration = int(dur_sec * sr // full_hop)
        duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))
        gen_kwargs["duration"] = duration

    with torch.no_grad():
        output = model(**gen_kwargs)

    wav = output.waveform.squeeze().detach().cpu().numpy()
    return sr, wav


# ─── High-level generation functions ──────────────────────────

def generate_tts(
    text: str,
    model_name: str,
    guidance_method: str,
    nfe: int,
    guidance_strength: float,
    seed: int,
    number_mode: str,
    enable_ssml: bool,
    do_fullwidth: bool,
    do_filter_symbols: bool,
    do_segment: bool,
    speed: int,
    volume: int,
    target_sr: int,
    audio_format: str,
    do_trim: bool,
    do_agc: bool,
    max_duration: float,
):
    if not text or not text.strip():
        raise gr.Error("请输入要合成的文本")

    # 文本预处理
    text = preprocess_text(text, do_fullwidth, do_filter_symbols)

    # SSML 或数字读法
    if enable_ssml and '<say-as' in text:
        text = parse_ssml(text, number_mode)
    else:
        text = auto_convert_numbers(text, number_mode)

    # 分段
    seg_info = ""
    if do_segment:
        segments = segment_text(text)
        if len(segments) > 1:
            seg_info = f"📋 总字数: {sum(len(s) for s in segments)} | 分{len(segments)}段: " + ", ".join(f"{len(s)}字" for s in segments)
    else:
        segments = [text]
        seg_info = f"📋 总字数: {len(text)} | 不分段"

    # 生成音频
    all_wav = []
    sample_rate = None
    for seg in segments:
        sr, wav = generate_tts_core(
            text=seg,
            model_name=model_name,
            guidance_method=guidance_method,
            nfe=nfe,
            guidance_strength=guidance_strength,
            seed=seed,
            max_duration_sec=max_duration,
        )
        sample_rate = sr
        all_wav.append(wav)

    if not all_wav:
        raise gr.Error("生成失败")

    wav = np.concatenate(all_wav) if len(all_wav) > 1 else all_wav[0]
    sr = sample_rate

    # 后处理
    wav = adjust_speed(wav, sr, speed)
    if do_trim:
        wav = trim_silence(wav)
    if do_agc:
        wav = apply_agc(wav)
    wav = adjust_volume(wav, volume)

    # 采样率转换
    if target_sr != sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return (sr, wav), seg_info


def generate_voice_clone(
    text: str,
    prompt_text: str,
    prompt_audio,
    model_name: str,
    guidance_method: str,
    nfe: int,
    guidance_strength: float,
    seed: int,
    number_mode: str,
    speed: int,
    volume: int,
    target_sr: int,
    audio_format: str,
    do_trim: bool,
    do_agc: bool,
    max_duration: float,
):
    if not text or not text.strip():
        raise gr.Error("请输入要合成的文本")
    if prompt_audio is None:
        raise gr.Error("请上传参考音频")
    if not prompt_text or not prompt_text.strip():
        raise gr.Error("请输入参考音频对应的文本")

    # 处理参考音频
    input_sr, audio_np = prompt_audio
    # Gradio 返回可能是 int16 或 float32
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32) / np.iinfo(np.int16).max
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=-1)
    # 归一化到 [-1, 1]
    peak = np.abs(audio_np).max()
    if peak > 1.0:
        audio_np = audio_np / peak
    elif peak == 0:
        raise gr.Error("参考音频为静音，请上传有效音频")
    if input_sr != 24000:
        pass  # Will resample after getting model SR below

    prompt_wav = torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(0)  # (1, 1, T) float32

    # 文本预处理
    text = auto_convert_numbers(preprocess_text(text), number_mode)

    # 先获取模型以确定采样率
    model, _ = get_model(model_name)
    model_sr = model.config.sampling_rate

    # 重采样参考音频到模型采样率
    if input_sr != model_sr:
        audio_np_resampled = librosa.resample(audio_np, orig_sr=input_sr, target_sr=model_sr)
        prompt_wav = torch.from_numpy(audio_np_resampled).float().unsqueeze(0).unsqueeze(0)

    # 克隆也支持分段
    segments = segment_text(text)
    if len(segments) > 1:
        seg_info = f"📋 克隆分{len(segments)}段: " + ", ".join(f"{len(s)}字" for s in segments)
    else:
        seg_info = f"📋 总字数: {len(text)}"

    all_wav = []
    for seg in segments:
        sr, wav = generate_tts_core(
            text=seg,
            model_name=model_name,
            guidance_method=guidance_method,
            nfe=nfe,
            guidance_strength=guidance_strength,
            seed=seed,
            prompt_audio_wav=prompt_wav,
            prompt_text=prompt_text,
            max_duration_sec=max_duration,
        )
        all_wav.append(wav)

    wav = np.concatenate(all_wav) if len(all_wav) > 1 else all_wav[0]

    # 后处理（克隆模式保持原始质量，不做语速调整）
    if do_trim:
        wav = trim_silence(wav)
    if do_agc:
        wav = apply_agc(wav)
    wav = adjust_volume(wav, volume)

    # 克隆模式不重采样，保持模型原始采样率
    return (sr, wav), seg_info


def generate_dialog(
    dialog_text: str,
    model_name: str,
    guidance_method: str,
    nfe: int,
    guidance_strength: float,
    base_seed: int,
    number_mode: str,
    speed: int,
    volume: int,
    target_sr: int,
    do_trim: bool,
    do_agc: bool,
    max_duration: float,
    role_a_mode: str, role_a_seed, role_a_audio, role_a_prompt_text: str,
    role_b_mode: str, role_b_seed, role_b_audio, role_b_prompt_text: str,
    role_c_mode: str, role_c_seed, role_c_audio, role_c_prompt_text: str,
):
    """多角色对话生成"""
    if not dialog_text or not dialog_text.strip():
        raise gr.Error("请输入对话文本")

    # 解析对话：每行 [X]文本
    lines = [l.strip() for l in dialog_text.strip().split('\n') if l.strip()]
    parsed = []  # [(role, text), ...]
    for line in lines:
        m = re.match(r'\[([ABCabc])\]\s*(.*)', line)
        if m:
            parsed.append((m.group(1).upper(), m.group(2).strip()))
        else:
            parsed.append(('A', line))

    if not parsed:
        raise gr.Error("没有有效的对话内容")

    # 角色配置
    roles = {
        'A': {'mode': role_a_mode, 'seed': int(role_a_seed) if role_a_seed else 100, 'audio': role_a_audio, 'text': role_a_prompt_text},
        'B': {'mode': role_b_mode, 'seed': int(role_b_seed) if role_b_seed else 200, 'audio': role_b_audio, 'text': role_b_prompt_text},
        'C': {'mode': role_c_mode, 'seed': int(role_c_seed) if role_c_seed else 300, 'audio': role_c_audio, 'text': role_c_prompt_text},
    }

    # 获取模型采样率
    model, _ = get_model(model_name)
    model_sr = model.config.sampling_rate

    # 预处理克隆角色的参考音频
    for role_id, role in roles.items():
        if role['mode'] == '克隆' and role['audio'] is not None:
            input_sr, audio_np = role['audio']
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32) / np.iinfo(np.int16).max
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=-1)
            peak = np.abs(audio_np).max()
            if peak > 1.0:
                audio_np = audio_np / peak
            if input_sr != model_sr:
                audio_np = librosa.resample(audio_np, orig_sr=input_sr, target_sr=model_sr)
            role['prompt_wav'] = torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(0)
        else:
            role['prompt_wav'] = None

    # 逐句生成
    all_wav = []
    info_lines = []
    for i, (role_id, text) in enumerate(parsed):
        if not text.strip():
            continue
        role = roles.get(role_id, roles['A'])
        text_clean = auto_convert_numbers(preprocess_text(text), number_mode)

        try:
            if role['mode'] == '克隆' and role['prompt_wav'] is not None and role['text']:
                sr, wav = generate_tts_core(
                    text=text_clean,
                    model_name=model_name,
                    guidance_method='apg',
                    nfe=nfe,
                    guidance_strength=guidance_strength,
                    seed=role['seed'],
                    prompt_audio_wav=role['prompt_wav'],
                    prompt_text=role['text'],
                    max_duration_sec=max_duration,
                )
            else:
                sr, wav = generate_tts_core(
                    text=text_clean,
                    model_name=model_name,
                    guidance_method=guidance_method,
                    nfe=nfe,
                    guidance_strength=guidance_strength,
                    seed=role['seed'],
                    max_duration_sec=max_duration,
                )
            all_wav.append(wav)
            mode_tag = "克隆" if role['mode'] == '克隆' else f"seed={role['seed']}"
            info_lines.append(f"{role_id}: {text[:20]}{'...' if len(text)>20 else ''} ({mode_tag})")
        except Exception as e:
            info_lines.append(f"{role_id}: ❌ {e}")

    if not all_wav:
        raise gr.Error("对话生成失败")

    wav = np.concatenate(all_wav) if len(all_wav) > 1 else all_wav[0]
    info = f"📋 共{len(parsed)}句对话\n" + "\n".join(info_lines)
    return (sr, wav), info


# ─── SSML Templates ───────────────────────────────────────────

# ─── GPU Monitor ────────────────────────────────────────────

def get_gpu_info():
    """获取 GPU 显存和使用率信息"""
    if not torch.cuda.is_available():
        return "❌ 无可用 GPU"
    try:
        import subprocess
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            parts = [p.strip() for p in r.stdout.strip().split(',')]
            util = parts[0]
            used = int(parts[1])
            total = int(parts[2])
            pct = used / total * 100 if total > 0 else 0
            return f"🎮 显存: {used}MB / {total}MB ({pct:.0f}%) | GPU利用率: {util}%"
        return "🎮 nvidia-smi 不可用"
    except Exception as e:
        return f"🎮 {e}"


def clear_vram():
    """清空显存（卸载所有缓存模型）"""
    cleared = []
    for name in list(MODEL_CACHE.keys()):
        model, tokenizer = MODEL_CACHE.pop(name)
        del model, tokenizer
        cleared.append(name)
    global _asr_model
    if _asr_model is not None:
        del _asr_model
        _asr_model = None
        cleared.append("SenseVoice ASR")
    torch.cuda.empty_cache()
    gc.collect()
    if cleared:
        return f"✅ 已卸载: {', '.join(cleared)}\n{get_gpu_info()}"
    return "⚠️ 没有已加载的模型\n" + get_gpu_info()


import gc

SSML_TEMPLATES = {
    "数字逐字读": '<say-as interpret-as="digits">12345</say-as>',
    "数值读法": '<say-as interpret-as="cardinal">12345</say-as>',
    "电话读法": '<say-as interpret-as="telephone">13800138000</say-as>',
    "日期读法": '<say-as interpret-as="date">20260425</say-as>',
    "金额读法": '<say-as interpret-as="currency">19.90</say-as>',
}


# ─── UI ───────────────────────────────────────────────────────

CSS = """
#main-tabs { max-width: 1200px; margin: auto; }
.speaker-card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 4px; }
.ssml-template-btn { font-size: 12px; padding: 4px 8px; }
textarea[data-testid="textbox"] { min-height: 60px !important; }
.gpu-bar { background: #f0f0f0; color: #333; padding: 8px 12px; border-radius: 6px; font-family: monospace; font-size: 13px; width: 100%; }
"""

with gr.Blocks(title="🐱 LongCat-AudioDiT TTS", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐱 LongCat-AudioDiT TTS — 全功能语音合成")
    gr.Markdown(
        "基于 [LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT) 的语音合成与声音克隆。"
        "支持 SSML、数字读法控制、语速/音量调节、多角色等功能。"
    )

    # ── 全局高级设置（放在底部 Accordion）──
    with gr.Accordion("⚙️ 全局设置", open=False):
        with gr.Row():
            guidance_method = gr.Radio(
                label="引导方法 (Guidance)",
                choices=["cfg", "apg"],
                value="cfg",
                info="声音克隆建议选 apg（自适应投影引导）",
            )
        with gr.Row():
            nfe = gr.Slider(label="扩散步数 (NFE Steps)", minimum=4, maximum=64, step=1, value=16,
                          info="步数越高质量越好，但越慢")
            guidance_strength = gr.Slider(
                label="引导强度 (Guidance Strength)", minimum=0.0, maximum=10.0, step=0.1, value=4.0,
                info="越高越贴合文本",
            )
        with gr.Row():
            seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=1024)
            randomize_seed = gr.Checkbox(label="随机 Seed", value=True)

        with gr.Row():
            number_mode = gr.Radio(
                label="数字读法模式",
                choices=["auto", "digits", "cardinal", "telephone", "date", "currency"],
                value="auto",
                info="auto=不转换，其余见说明。SSML中<say-as>优先级更高",
            )

        with gr.Row():
            speed = gr.Slider(label="语速 (Speed)", minimum=0, maximum=100, step=1, value=50,
                            info="50=原始，0=最慢，100=最快")
            volume = gr.Slider(label="音量 (Volume)", minimum=0, maximum=100, step=1, value=50,
                             info="50=原始，100=最大")
            max_duration = gr.Slider(label="最大时长(秒)", minimum=5, maximum=60, step=1, value=30)

        with gr.Row():
            target_sr = gr.Dropdown(
                label="输出采样率",
                choices=[8000, 16000, 24000, 32000, 44100, 48000],
                value=24000,
            )
            audio_format = gr.Dropdown(
                label="音频格式",
                choices=["wav", "mp3"],
                value="wav",
            )

        with gr.Row():
            do_trim = gr.Checkbox(label="去除首尾静音", value=True)
            do_agc = gr.Checkbox(label="自动增益 (AGC)", value=False)
            do_fullwidth = gr.Checkbox(label="全角→半角数字", value=True)
            do_filter_symbols = gr.Checkbox(label="过滤特殊符号", value=True)
            do_segment = gr.Checkbox(label="超长文本自动分段", value=True)

    # ── 主 Tabs ──
    with gr.Tabs(elem_id="main-tabs"):

        # ═══ Tab 1: TTS 合成 ═══
        with gr.Tab("🎤 TTS 合成"):
            with gr.Row():
                with gr.Column(scale=2):
                    enable_ssml = gr.Checkbox(label="启用 SSML", value=False)
                    tts_text = gr.Textbox(
                        label="合成文本",
                        lines=6,
                        placeholder="输入要合成的文本... 或开启SSML后使用<say-as>标签",
                    )
                    # SSML 快速模板
                    with gr.Row(visible=False) as ssml_row:
                        for name, tpl in SSML_TEMPLATES.items():
                            gr.Button(name, elem_classes="ssml-template-btn").click(
                                lambda t=tpl: t,
                                outputs=tts_text,
                            )
                    enable_ssml.change(lambda v: gr.Row(visible=v), enable_ssml, ssml_row)

                    with gr.Row():
                        tts_model = gr.Radio(label="模型", choices=["1B", "3.5B"], value="1B")
                    tts_btn = gr.Button("🎤 生成", variant="primary", size="lg")

                with gr.Column(scale=1):
                    tts_output = gr.Audio(label="生成结果", type="numpy")
                    tts_info = gr.Textbox(label="状态", visible=True, interactive=False, show_copy_button=True)
                    with gr.Row():
                        tts_gpu = gr.Markdown(value="🎮 加载中...", elem_classes="gpu-bar")
                    with gr.Row():
                        tts_clear_btn = gr.Button("🗑️ 清空显存", size="sm")
                        tts_clear_btn.click(clear_vram, outputs=tts_gpu)

            gr.Examples(
                examples=[
                    ["今天晴暖转阴雨，空气质量优至良，空气相对湿度较低。"],
                    ["The quick brown fox jumps over the lazy dog."],
                    ["大江东去浪淘尽，千古风流人物。故垒西边，人道是，三国周郎赤壁。"],
                    ['电话号码是<say-as interpret-as="telephone">13800138000</say-as>，请记录。'],
                    ['今天是<say-as interpret-as="date">20260425</say-as>，价格<say-as interpret-as="currency">19.90</say-as>'],
                ],
                inputs=tts_text,
            )

        # ═══ Tab 2: 声音克隆 ═══
        with gr.Tab("🎭 声音克隆"):
            with gr.Row():
                with gr.Column():
                    vc_prompt_audio = gr.Audio(label="参考音频（3-15秒，wav/mp3）", type="numpy")
                    vc_prompt_text = gr.Textbox(
                        label="参考音频文本（上传音频后自动识别）",
                        lines=2,
                        placeholder="上传参考音频后自动填充，也可手动修改...",
                    )
                    vc_asr_btn = gr.Button("🔤 自动识别音频文本", size="sm")
                    vc_asr_btn.click(
                        fn=transcribe_audio,
                        inputs=[vc_prompt_audio],
                        outputs=[vc_prompt_text],
                    )
                    vc_prompt_audio.change(
                        fn=transcribe_audio,
                        inputs=[vc_prompt_audio],
                        outputs=[vc_prompt_text],
                    )
                    vc_text = gr.Textbox(
                        label="合成文本",
                        lines=4,
                        placeholder="输入要用克隆声音合成的文本...",
                    )
                    with gr.Row():
                        vc_model = gr.Radio(label="模型", choices=["1B", "3.5B"], value="1B")
                    vc_btn = gr.Button("🎭 克隆生成", variant="primary", size="lg")

                with gr.Column():
                    vc_output = gr.Audio(label="生成结果", type="numpy")
                    vc_info = gr.Textbox(label="状态", interactive=False, show_copy_button=True)
                    gr.Markdown("### 使用说明\n"
                                "1. 上传 3-15 秒参考音频\n"
                                "2. 输入参考音频文字\n"
                                "3. 输入合成文本\n"
                                "4. 建议 **APG + 3.5B + 32步** 效果最佳\n"
                                "5. 参考音频越清晰，克隆效果越好")
                    with gr.Row():
                        vc_gpu = gr.Markdown(value="🎮 加载中...", elem_classes="gpu-bar")
                    with gr.Row():
                        vc_clear_btn = gr.Button("🗑️ 清空显存", size="sm")
                        vc_clear_btn.click(clear_vram, outputs=vc_gpu)

        # ═══ Tab 3: 多角色对话 ═══
        with gr.Tab("🎭 多角色对话"):
            gr.Markdown("### 角色定义\n定义最多3个角色的音色。\n\n⚠️ **LongCat 不支持通过seed控制音色**，纯TTS模式每次生成的音色都是随机的。\n**建议用克隆模式**：上传3-5秒参考音频来定义角色音色，这样同一角色在所有台词中音色一致。")
            with gr.Row():
                # 角色 A
                with gr.Column(scale=1):
                    gr.Markdown("**🅰️ 角色 A**")
                    role_a_mode = gr.Radio(label="音色来源", choices=["随机", "克隆"], value="克隆")
                    role_a_seed = gr.Number(label="Seed", value=100, precision=0, visible=True)
                    role_a_audio = gr.Audio(label="参考音频", type="numpy", visible=False)
                    role_a_text = gr.Textbox(label="参考音频文本", visible=False, placeholder="参考音频对应文字...")
                    role_a_test_btn = gr.Button("🔊 试听A", size="sm")
                    role_a_test_out = gr.Audio(label="试听", type="numpy")
                # 角色 B
                with gr.Column(scale=1):
                    gr.Markdown("**🅱️ 角色 B**")
                    role_b_mode = gr.Radio(label="音色来源", choices=["随机", "克隆"], value="克隆")
                    role_b_seed = gr.Number(label="Seed", value=200, precision=0, visible=True)
                    role_b_audio = gr.Audio(label="参考音频", type="numpy", visible=False)
                    role_b_text = gr.Textbox(label="参考音频文本", visible=False, placeholder="参考音频对应文字...")
                    role_b_test_btn = gr.Button("🔊 试听B", size="sm")
                    role_b_test_out = gr.Audio(label="试听", type="numpy")
                # 角色 C
                with gr.Column(scale=1):
                    gr.Markdown("**🅲 角色 C**")
                    role_c_mode = gr.Radio(label="音色来源", choices=["随机", "克隆"], value="克隆")
                    role_c_seed = gr.Number(label="Seed", value=300, precision=0, visible=True)
                    role_c_audio = gr.Audio(label="参考音频", type="numpy", visible=False)
                    role_c_text = gr.Textbox(label="参考音频文本", visible=False, placeholder="参考音频对应文字...")
                    role_c_test_btn = gr.Button("🔊 试听C", size="sm")
                    role_c_test_out = gr.Audio(label="试听", type="numpy")

            gr.Markdown("### 对话输入\n用 `[A]` `[B]` `[C]` 标记每句台词的角色，每行一句。无标记默认为A。")
            batch_text = gr.Textbox(
                label="对话文本",
                lines=10,
                placeholder="[A]你好，今天天气真不错\n[B]是啊，我们去公园吧\n[A]好主意！\n[C]我也想去！\n[B]那我们出发吧",
            )
            with gr.Row():
                batch_model = gr.Radio(label="模型", choices=["1B", "3.5B"], value="1B")
            batch_btn = gr.Button("🎭 生成对话", variant="primary", size="lg")
            batch_output = gr.Audio(label="对话结果", type="numpy")
            batch_info = gr.Textbox(label="状态", interactive=False, show_copy_button=True)
            with gr.Row():
                batch_gpu = gr.Markdown(value="🎮 加载中...", elem_classes="gpu-bar")
            with gr.Row():
                batch_clear_btn = gr.Button("🗑️ 清空显存", size="sm")
                batch_clear_btn.click(clear_vram, outputs=batch_gpu)

            # 角色模式切换
            def toggle_role_mode(mode):
                is_clone = mode == "克隆"
                return gr.Number(visible=not is_clone), gr.Audio(visible=is_clone), gr.Textbox(visible=is_clone)
            role_a_mode.change(toggle_role_mode, role_a_mode, [role_a_seed, role_a_audio, role_a_text])
            role_b_mode.change(toggle_role_mode, role_b_mode, [role_b_seed, role_b_audio, role_b_text])
            role_c_mode.change(toggle_role_mode, role_c_mode, [role_c_seed, role_c_audio, role_c_text])

            # 试听按钮
            def test_role_voice(mode, seed_val, audio, text, model_name):
                try:
                    if mode == "克隆":
                        if audio is None:
                            raise gr.Error("请上传参考音频")
                        if not text or not text.strip():
                            raise gr.Error("请输入参考音频文本")
                        (result, _) = generate_voice_clone(
                            "这是试听音频，测试一下音色效果。", text, audio, model_name,
                            "apg", 16, 4.0, int(seed_val) if seed_val else 1024,
                            "auto", 50, 50, 24000, "wav", True, False, 30.0,
                        )
                        return result
                    else:
                        (result, _) = generate_tts(
                            "这是试听音频，测试一下音色效果。", model_name,
                            "cfg", 16, 4.0, int(seed_val) if seed_val else 1024,
                            "auto", False, True, True, False,
                            50, 50, 24000, "wav", True, False, 30.0,
                        )
                        return result
                except Exception as e:
                    raise gr.Error(str(e))

            role_a_test_btn.click(fn=test_role_voice,
                inputs=[role_a_mode, role_a_seed, role_a_audio, role_a_text, batch_model],
                outputs=role_a_test_out)
            role_b_test_btn.click(fn=test_role_voice,
                inputs=[role_b_mode, role_b_seed, role_b_audio, role_b_text, batch_model],
                outputs=role_b_test_out)
            role_c_test_btn.click(fn=test_role_voice,
                inputs=[role_c_mode, role_c_seed, role_c_audio, role_c_text, batch_model],
                outputs=role_c_test_out)

        # ═══ Tab 4: SSML 编辑器 ═══
        with gr.Tab("📝 SSML 编辑器"):
            gr.Markdown(
                "### SSML 标签说明\n"
                "| 标签 | 说明 | 示例 |\n"
                "|---|---|---|\n"
                "| `digits` | 逐字读 | 123 → 一二三 |\n"
                "| `cardinal` | 数值读 | 123 → 一百二十三 |\n"
                "| `telephone` | 电话读法 | 138 → 幺三八 |\n"
                "| `date` | 日期读法 | 20260425 → 二零二六年四月二十五日 |\n"
                "| `currency` | 金额读法 | 19.9 → 十九点九元 |\n"
            )
            ssml_input = gr.Textbox(
                label="SSML 输入",
                lines=8,
                value='<speak>\n电话号码是<say-as interpret-as="telephone">13800138000</say-as>\n今天是<say-as interpret-as="date">20260425</say-as>\n价格<say-as interpret-as="currency">99.80</say-as>\n</speak>',
            )
            with gr.Row():
                ssml_model = gr.Radio(label="模型", choices=["1B", "3.5B"], value="1B")
            ssml_btn = gr.Button("📝 SSML 合成", variant="primary")
            ssml_output = gr.Audio(label="生成结果", type="numpy")
            with gr.Row():
                ssml_gpu = gr.Markdown(value="🎮 加载中...", elem_classes="gpu-bar")
            with gr.Row():
                ssml_clear_btn = gr.Button("🗑️ 清空显存", size="sm")
                ssml_clear_btn.click(clear_vram, outputs=ssml_gpu)

        # ═══ Tab 5: 模型管理 ═══
        with gr.Tab("🔧 模型管理"):
            gr.Markdown("### 当前已加载模型")
            model_status = gr.Textbox(label="模型状态", interactive=False, lines=5)
            refresh_btn = gr.Button("🔄 刷新状态")

            def refresh_model_status():
                lines = []
                for name in ["1B", "3.5B"]:
                    if name in MODEL_CACHE:
                        model, tokenizer = MODEL_CACHE[name]
                        params = sum(p.numel() for p in model.parameters()) / 1e9
                        lines.append(f"✅ {name}: 已加载 ({params:.1f}B 参数)")
                    else:
                        lines.append(f"⬜ {name}: 未加载（首次使用时自动加载）")
                return '\n'.join(lines)

            refresh_btn.click(refresh_model_status, outputs=model_status)

    # ── Event handlers ────────────────────────────────────────

    def get_seed_value(randomize, seed_val):
        if randomize:
            return int(np.random.default_rng().integers(0, MAX_SEED))
        return seed_val

    # TTS 合成
    def tts_wrapper(text, model, gm, nfe_val, gs, seed_val, rs, nm, ssml, fw, fs, seg, spd, vol, tsr, af, trim, agc, md):
        try:
            (audio, seg_info) = generate_tts(text, model, gm, nfe_val, gs, seed_val, nm, ssml, fw, fs, seg, spd, vol, tsr, af, trim, agc, md)
            return audio, f"✅ 生成成功 | 模型: {model} | Seed: {seed_val}\n{seg_info}"
        except Exception as e:
            return None, f"❌ 错误: {str(e)}"

    tts_btn.click(fn=get_seed_value, inputs=[randomize_seed, seed], outputs=seed, queue=False).then(
        fn=tts_wrapper,
        inputs=[tts_text, tts_model, guidance_method, nfe, guidance_strength, seed,
                randomize_seed, number_mode, enable_ssml, do_fullwidth, do_filter_symbols,
                do_segment, speed, volume, target_sr, audio_format, do_trim, do_agc, max_duration],
        outputs=[tts_output, tts_info],
    )

    # 声音克隆
    def vc_wrapper(text, pt, pa, model, gm, nfe_val, gs, seed_val, rs, nm, spd, vol, tsr, af, trim, agc, md):
        try:
            (audio, seg_info) = generate_voice_clone(text, pt, pa, model, gm, nfe_val, gs, seed_val, nm, spd, vol, tsr, af, trim, agc, md)
            return audio, f"✅ 克隆成功 | 模型: {model}\n{seg_info}"
        except Exception as e:
            raise gr.Error(str(e))

    vc_btn.click(fn=get_seed_value, inputs=[randomize_seed, seed], outputs=seed, queue=False).then(
        fn=vc_wrapper,
        inputs=[vc_text, vc_prompt_text, vc_prompt_audio, vc_model,
                guidance_method, nfe, guidance_strength, seed, randomize_seed,
                number_mode, speed, volume, target_sr, audio_format, do_trim, do_agc, max_duration],
        outputs=[vc_output, vc_info],
    )

    # 多角色对话
    def batch_wrapper(dialog_text, model, gm, nfe_val, gs, seed_val, rs, nm, spd, vol, tsr, trim, agc, md,
                      ra_mode, ra_seed, ra_audio, ra_text,
                      rb_mode, rb_seed, rb_audio, rb_text,
                      rc_mode, rc_seed, rc_audio, rc_text):
        try:
            (audio, info) = generate_dialog(
                dialog_text, model, gm, nfe_val, gs, seed_val, nm, spd, vol, tsr, trim, agc, md,
                ra_mode, ra_seed, ra_audio, ra_text,
                rb_mode, rb_seed, rb_audio, rb_text,
                rc_mode, rc_seed, rc_audio, rc_text,
            )
            return audio, f"✅ 对话生成完成\n{info}"
        except Exception as e:
            return None, f"❌ 错误: {str(e)}"

    batch_btn.click(fn=get_seed_value, inputs=[randomize_seed, seed], outputs=seed, queue=False).then(
        fn=batch_wrapper,
        inputs=[batch_text, batch_model, guidance_method, nfe, guidance_strength, seed, randomize_seed,
                number_mode, speed, volume, target_sr, do_trim, do_agc, max_duration,
                role_a_mode, role_a_seed, role_a_audio, role_a_text,
                role_b_mode, role_b_seed, role_b_audio, role_b_text,
                role_c_mode, role_c_seed, role_c_audio, role_c_text],
        outputs=[batch_output, batch_info],
    )

    # SSML 合成
    def ssml_wrapper(text, model, gm, nfe_val, gs, seed_val, rs, nm, spd, vol, tsr, trim, agc, md):
        try:
            (audio, _) = generate_tts(
                text, model, gm, nfe_val, gs, seed_val, nm, True, True, False, False,
                spd, vol, tsr, 'wav', trim, agc, md,
            )
            return audio
        except Exception as e:
            raise gr.Error(str(e))

    ssml_btn.click(fn=get_seed_value, inputs=[randomize_seed, seed], outputs=seed, queue=False).then(
        fn=ssml_wrapper,
        inputs=[ssml_input, ssml_model, guidance_method, nfe, guidance_strength, seed, randomize_seed,
                number_mode, speed, volume, target_sr, do_trim, do_agc, max_duration],
        outputs=[ssml_output],
    )

    # 页面加载时刷新模型状态
    demo.load(fn=refresh_model_status, outputs=model_status)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
