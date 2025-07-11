import soundfile as sf
import torch
import torchaudio
import numpy as np
import pyworld
import matplotlib.pyplot as plt

# 1. 读取音频
wav_path = 'data/1000185_1b8014bb.wav'
txt_path = 'data/1000185_1b8014bb.text'
waveform, sample_rate = sf.read(wav_path)
if waveform.ndim > 1:
    waveform = waveform[0]  # 只取第一通道
x = waveform.astype(np.float64)

# 2. 读取文本并分音节（每个汉字为一个音节）
with open(txt_path, 'r', encoding='utf-8') as f:
    text = f.read().strip().replace(' ', '').replace('\n', '')
syllables = list(text)  # 简单假设每个汉字为一个音节

# 3. VAD
waveform_torch = torch.from_numpy(waveform).unsqueeze(0)
vad = torchaudio.transforms.Vad(sample_rate=sample_rate)
vad_waveform = vad(waveform_torch)
vad_mask = (vad_waveform.abs().sum(dim=0) > 0).numpy()
vad_segments = []
in_speech = False
for i, flag in enumerate(vad_mask):
    t = i / sample_rate
    if flag and not in_speech:
        start = t
        in_speech = True
    elif not flag and in_speech:
        end = t
        vad_segments.append((start, end))
        in_speech = False
if in_speech:
    vad_segments.append((start, len(vad_mask) / sample_rate))

# 4. F0提取
frame_period = 10.0  # ms
_f0, t = pyworld.dio(x, sample_rate, frame_period=frame_period)
f0 = pyworld.stonemask(x, _f0, t, sample_rate)
f0_times = t  # 单位：秒

# 5. 五度标记法
def f0_to_scale(f0_value):
    if f0_value <= 0:
        return '—'
    scale = 12 * np.log2(f0_value / 261.63)
    if scale < -6:
        return '1'
    elif scale < -3:
        return '2'
    elif scale < 0:
        return '3'
    elif scale < 3:
        return '4'
    elif scale < 6:
        return '5'
    elif scale < 9:
        return '6'
    else:
        return '7'

scale_marks = [f0_to_scale(f) for f in f0]

# 6. 音节与F0帧简单对齐（跨所有VAD段等分）
vad_mask_f0 = np.zeros_like(f0_times, dtype=bool)
for start, end in vad_segments:
    vad_mask_f0 |= (f0_times >= start) & (f0_times < end)
f0_idxs = np.where(vad_mask_f0)[0]
n_syll = len(syllables)
n_f0 = len(f0_idxs)
syllable_f0_indices = np.array_split(f0_idxs, n_syll)
syllable_times = []
syllable_scales = []
for idxs in syllable_f0_indices:
    if len(idxs) == 0:
        syllable_times.append((None, None))
        syllable_scales.append('—')
    else:
        start_time = f0_times[idxs[0]]
        end_time = f0_times[idxs[-1]]
        mean_f0 = np.mean(f0[idxs])
        syllable_times.append((start_time, end_time))
        syllable_scales.append(f0_to_scale(mean_f0))

# 输出音节与时间帧的对应关系
print("音节\t起始时间(s)\t结束时间(s)\t五度标记")
for s, (start, end), scale in zip(syllables, syllable_times, syllable_scales):
    if start is not None and end is not None:
        print(f"{s}\t{start:.2f}\t\t{end:.2f}\t\t{scale}")
    else:
        print(f"{s}\tNone\t\tNone\t\t{scale}")

print(f"VAD段数量: {len(vad_segments)}")
print(f"F0帧数量: {len(f0_times)}")
print(f"音节数量: {len(syllables)}")

# 7. 可视化
plt.figure(figsize=(14, 6))
plt.plot(f0_times, f0, label='F0 (Hz)')
for (start, end) in vad_segments:
    plt.axvspan(start, end, color='orange', alpha=0.2, label='VAD Segment' if start == vad_segments[0][0] else "")
plt.xlabel('Time (s)')
plt.ylabel('F0 (Hz)')
plt.title('F0 Track with VAD Segments and Syllable Matching')

# 标注五度标记
for t_val, f0_val, mark in zip(f0_times, f0, scale_marks):
    if mark != '—':
        plt.text(t_val, f0_val+5, mark, fontsize=8, color='blue', ha='center', va='bottom')

plt.legend()
plt.tight_layout()
plt.savefig('demo.png', dpi=300)