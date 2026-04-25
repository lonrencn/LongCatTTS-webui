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


def segment_text(text: str, max_chars: int = 512) -> list:
    """超长文本自动分段"""
    if len(text) <= max_chars:
        return [text]
    # 按句号/问号/感叹号分段
    sentences = re.split(r'([。？！\.\?\!])', text)
    segments = []
    current = ''
    for i in range(0, len(sentences) - 1, 2):
        s = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else '')
        if len(current) + len(s) > max_chars:
            if current:
                segments.append(current)
            current = s
        else:
            current += s
    if len(sentences) % 2 == 1 and sentences[-1]:
        current += sentences[-1]
    if current:
        segments.append(current)
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
    if do_segment:
        segments = segment_text(text)
    else:
        segments = [text]

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

    return (sr, wav)


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
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=-1)
    audio_np = audio_np.astype(np.float32)
    if np.abs(audio_np).max() > 1.0:
        audio_np = audio_np / np.abs(audio_np).max()
    if input_sr != 24000:
        pass  # Will resample after getting model SR below

    prompt_wav = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)

    # 文本预处理
    text = auto_convert_numbers(preprocess_text(text), number_mode)

    # 先获取模型以确定采样率
    model, _ = get_model(model_name)
    model_sr = model.config.sampling_rate

    # 重采样参考音频到模型采样率
    if input_sr != model_sr:
        audio_np_resampled = librosa.resample(audio_np, orig_sr=input_sr, target_sr=model_sr)
        prompt_wav = torch.from_numpy(audio_np_resampled).unsqueeze(0).unsqueeze(0)

    sr, wav = generate_tts_core(
        text=text,
        model_name=model_name,
        guidance_method=guidance_method,
        nfe=nfe,
        guidance_strength=guidance_strength,
        seed=seed,
        prompt_audio_wav=prompt_wav,
        prompt_text=prompt_text,
        max_duration_sec=max_duration,
    )

    # 后处理
    wav = adjust_speed(wav, sr, speed)
    if do_trim:
        wav = trim_silence(wav)
    if do_agc:
        wav = apply_agc(wav)
    wav = adjust_volume(wav, volume)

    if target_sr != sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return (sr, wav)


def generate_batch(
    texts: str,
    model_name: str,
    guidance_method: str,
    nfe: int,
    guidance_strength: float,
    seed: int,
    number_mode: str,
    speed: int,
    volume: int,
    target_sr: int,
    do_trim: bool,
    do_agc: bool,
    max_duration: float,
):
    """批量合成：每行一条文本"""
    if not texts or not texts.strip():
        raise gr.Error("请输入要合成的文本（每行一条）")

    lines = [l.strip() for l in texts.strip().split('\n') if l.strip()]
    results = []
    for i, line in enumerate(lines):
        try:
            sr, wav = generate_tts(
                text=line,
                model_name=model_name,
                guidance_method=guidance_method,
                nfe=nfe,
                guidance_strength=guidance_strength,
                seed=seed + i,
                number_mode=number_mode,
                enable_ssml=False,
                do_fullwidth=True,
                do_filter_symbols=True,
                do_segment=False,
                speed=speed,
                volume=volume,
                target_sr=target_sr,
                audio_format='wav',
                do_trim=do_trim,
                do_agc=do_agc,
                max_duration=max_duration,
            )
            results.append((sr, wav))
        except Exception as e:
            results.append(None)

    # 返回最后一条作为预览
    valid = [(sr, wav) for r in results if r is not None for sr, wav in [r]]
    if valid:
        return valid[-1]
    raise gr.Error("批量生成全部失败")


# ─── SSML Templates ───────────────────────────────────────────

# ─── GPU Monitor ────────────────────────────────────────────

def get_gpu_info():
    """获取 GPU 显存和使用率信息"""
    if not torch.cuda.is_available():
        return "❌ 无可用 GPU"
    try:
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_mem / 1024**3
        util = 0
        try:
            import subprocess
            r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                parts = r.stdout.strip().split(',')
                util = int(parts[0])
                used_mb = int(parts[1])
                total_mb = int(parts[2])
                return f"🎮 {used_mb}MB/{total_mb}MB ({used_mb/total_mb*100:.0f}%) | GPU: {util}%"
        except Exception:
            pass
        return f"🎮 {allocated:.1f}G/{total:.1f}G ({allocated/total*100:.0f}%) | GPU: {util}%"
    except Exception as e:
        return f"❌ {e}"


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
"""

with gr.Blocks(title="🐱 LongCat-AudioDiT TTS", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐱 LongCat-AudioDiT TTS — 全功能语音合成")
    gr.Markdown(
        "基于 [LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT) 的语音合成与声音克隆。"
        "支持 SSML、数字读法控制、语速/音量调节、批量合成等功能。"
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

                with gr.Row():
                    tts_gpu = gr.Textbox(interactive=False, show_label=False, container=False)
                    tts_clear_btn = gr.Button("🗑️ 清空显存", size="sm")
                    tts_clear_btn.click(clear_vram, outputs=tts_gpu)

                with gr.Column(scale=1):
                    tts_output = gr.Audio(label="生成结果", type="numpy")
                    tts_info = gr.Textbox(label="状态", visible=True, interactive=False)

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

                with gr.Row():
                    vc_gpu = gr.Textbox(interactive=False, show_label=False, container=False)
                    vc_clear_btn = gr.Button("🗑️ 清空显存", size="sm")
                    vc_clear_btn.click(clear_vram, outputs=vc_gpu)

                with gr.Column():
                    vc_output = gr.Audio(label="生成结果", type="numpy")
                    gr.Markdown(
                        "### 使用说明\n"
                        "1. 上传一段 3-15 秒的参考音频\n"
                        "2. 输入参考音频对应的文字\n"
                        "3. 输入想要合成的文本\n"
                        "4. 建议：APG引导 + 3.5B模型效果最佳\n"
                        "5. 参考音频越清晰，克隆效果越好"
                    )

        # ═══ Tab 3: 批量合成 ═══
        with gr.Tab("📦 批量合成"):
            with gr.Row():
                with gr.Column():
                    batch_text = gr.Textbox(
                        label="批量文本（每行一条）",
                        lines=10,
                        placeholder="第一行文本\n第二行文本\n第三行文本...",
                    )
                    with gr.Row():
                        batch_model = gr.Radio(label="模型", choices=["1B", "3.5B"], value="1B")
                    batch_btn = gr.Button("📦 批量生成", variant="primary")
                with gr.Row():
                    batch_gpu = gr.Textbox(interactive=False, show_label=False, container=False)
                    batch_clear_btn = gr.Button("🗑️ 清空显存", size="sm")
                    batch_clear_btn.click(clear_vram, outputs=batch_gpu)
                with gr.Column():
                    batch_output = gr.Audio(label="最后一条预览", type="numpy")
                    batch_info = gr.Textbox(label="状态", interactive=False)

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
            with gr.Row():
                ssml_gpu = gr.Textbox(interactive=False, show_label=False, container=False)
                ssml_clear_btn = gr.Button("🗑️ 清空显存", size="sm")
                ssml_clear_btn.click(clear_vram, outputs=ssml_gpu)
            ssml_output = gr.Audio(label="生成结果", type="numpy")

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
            result = generate_tts(text, model, gm, nfe_val, gs, seed_val, nm, ssml, fw, fs, seg, spd, vol, tsr, af, trim, agc, md)
            return result, f"✅ 生成成功 | 模型: {model} | Seed: {seed_val}"
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
            result = generate_voice_clone(text, pt, pa, model, gm, nfe_val, gs, seed_val, nm, spd, vol, tsr, af, trim, agc, md)
            return result
        except Exception as e:
            raise gr.Error(str(e))

    vc_btn.click(fn=get_seed_value, inputs=[randomize_seed, seed], outputs=seed, queue=False).then(
        fn=vc_wrapper,
        inputs=[vc_text, vc_prompt_text, vc_prompt_audio, vc_model,
                guidance_method, nfe, guidance_strength, seed, randomize_seed,
                number_mode, speed, volume, target_sr, audio_format, do_trim, do_agc, max_duration],
        outputs=[vc_output],
    )

    # 批量合成
    def batch_wrapper(texts, model, gm, nfe_val, gs, seed_val, rs, nm, spd, vol, tsr, trim, agc, md):
        try:
            result = generate_batch(texts, model, gm, nfe_val, gs, seed_val, nm, spd, vol, tsr, 'wav', trim, agc, md)
            count = len([l for l in texts.strip().split('\n') if l.strip()])
            return result, f"✅ 已生成 {count} 条音频（预览最后一条）"
        except Exception as e:
            return None, f"❌ 错误: {str(e)}"

    batch_btn.click(fn=get_seed_value, inputs=[randomize_seed, seed], outputs=seed, queue=False).then(
        fn=batch_wrapper,
        inputs=[batch_text, batch_model, guidance_method, nfe, guidance_strength, seed, randomize_seed,
                number_mode, speed, volume, target_sr, do_trim, do_agc, max_duration],
        outputs=[batch_output, batch_info],
    )

    # SSML 合成
    def ssml_wrapper(text, model, gm, nfe_val, gs, seed_val, rs, nm, spd, vol, tsr, trim, agc, md):
        try:
            result = generate_tts(
                text, model, gm, nfe_val, gs, seed_val, nm, True, True, False, False,
                spd, vol, tsr, 'wav', trim, agc, md,
            )
            return result
        except Exception as e:
            raise gr.Error(str(e))

    ssml_btn.click(fn=get_seed_value, inputs=[randomize_seed, seed], outputs=seed, queue=False).then(
        fn=ssml_wrapper,
        inputs=[ssml_input, ssml_model, guidance_method, nfe, guidance_strength, seed, randomize_seed,
                number_mode, speed, volume, target_sr, do_trim, do_agc, max_duration],
        outputs=[ssml_output],
    )

    # 页面加载时刷新模型状态
    # GPU 状态定时刷新
    try:
        timer = gr.Timer(value=15)
        for gpu_comp in [tts_gpu, vc_gpu, batch_gpu, ssml_gpu]:
            timer.tick(fn=get_gpu_info, outputs=gpu_comp)
    except Exception:
        pass

    # 页面加载时初始化
    demo.load(fn=lambda: (refresh_model_status(), get_gpu_info(), get_gpu_info(), get_gpu_info(), get_gpu_info()),
              outputs=[model_status, tts_gpu, vc_gpu, batch_gpu, ssml_gpu])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
