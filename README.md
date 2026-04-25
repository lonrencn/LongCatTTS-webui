# LongCat-AudioDiT: High-Fidelity Diffusion Text-to-Speech in the Waveform Latent Space

给LongCat-AudioDiT用龙虾写了一个webui，大家看看能不能用上~
---

## 🎤 全功能 WebUI（本项目增强）

在官方代码基础上，新增了全功能 Gradio WebUI（`webui.py`），支持：

### 功能特性
- **TTS 合成**：基础文本转语音
- **声音克隆**：零样本声音克隆（上传参考音频）
- **批量合成**：多条文本逐行批量生成
- **SSML 编辑器**：支持 `<say-as>` 标签控制数字读法
- **模型管理**：1B / 3.5B 模型切换，查看加载状态

### 全局设置
- 引导方法：CFG / APG（自适应投影引导）
- 扩散步数（NFE）：4-64
- 引导强度：0-10
- 数字读法模式：自动 / 逐字读 / 数值读 / 电话读法 / 日期读法 / 金额读法
- 语速 / 音量调节
- 输出采样率：8k / 16k / 24k / 32k / 44.1k / 48k
- 音频格式：WAV / MP3
- 静音修剪、自动增益 (AGC)
- 超长文本自动分段（≤512字符/段）
- 全角→半角自动转换、特殊符号过滤

### 模型下载

模型文件不包含在仓库中，需单独下载：

**方式一：HuggingFace（推荐，国内可能较慢）**
\`\`\`bash
# 1B 模型
huggingface-cli download meituan-longcat/LongCat-AudioDiT-1B

# 3.5B 模型
huggingface-cli download meituan-longcat/LongCat-AudioDiT-3.5B
\`\`\`

**方式二：ModelScope（国内推荐）**
\`\`\`bash
pip install modelscope

# 1B 模型
python -c "from modelscope import snapshot_download; snapshot_download(meituan-longcat/LongCat-AudioDiT-1B, cache_dir=~/.cache/huggingface/hub)"

# 3.5B 模型
python -c "from modelscope import snapshot_download; snapshot_download(meituan-longcat/LongCat-AudioDiT-3.5B, cache_dir=~/.cache/huggingface/hub)"
\`\`\`

**方式三：手动下载**
- 1B: https://huggingface.co/meituan-longcat/LongCat-AudioDiT-1B
- 3.5B: https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B
- ModelScope 3.5B: https://modelscope.cn/models/meituan-longcat/LongCat-AudioDiT-3.5B

下载后放到 `~/.cache/huggingface/hub/` 目录下即可。

另外需要 `google/umt5-base` tokenizer（首次运行自动下载，或手动下载）：
\`\`\`bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(google/umt5-base)"
\`\`\`

### 启动
\`\`\`bash
conda activate longcat  # 或你的环境名
PYTHONPATH=. python webui.py
# 访问 http://localhost:7860
\`\`\`

### 环境要求
- Python 3.10+
- CUDA GPU（推荐 24GB+ 显存用于 3.5B 模型）
- PyTorch 2.0+
- 见 requirements.txt

### 显存参考
| 模型 | 显存占用 | 推荐 GPU |
|------|---------|---------|
| 1B   | ~8 GB   | RTX 3060 12G+ |
| 3.5B | ~17 GB  | RTX 3090 24G+ |
| 1B + 3.5B 同时 | ~25 GB | RTX 4090 24G |
