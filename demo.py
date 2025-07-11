import numpy as np
import soundfile as sf
import parselmouth
import matplotlib.pyplot as plt

# 配置参数
wav_path = 'data/1000185_306eba96.wav'
txt_path = 'data/1000185_306eba96.text'
sex = 'male'  # 'male', 'female', 'child'

# F0范围
F0_RANGE = {
    'male':   (80, 190),
    'female': (165, 255),
    'child':  (250, 400),
    'other':  (50, 500)
}
f0_min, f0_max = F0_RANGE.get(sex, F0_RANGE['other'])

# Formant范围
F1_MIN, F1_MAX = 200, 1000
F2_MIN, F2_MAX = 600, 3000

# 合并有声段最小间隔
MERGE_MIN_GAP = 0.05

# 非线性音节分配温度
SYLLABLE_TEMP = 0.8

# 加载音频与文本
signal, sample_rate = sf.read(wav_path)
if signal.ndim > 1:
    signal = signal.mean(axis=1)

with open(txt_path, 'r', encoding='utf-8') as f:
    text = f.read().strip().replace(' ', '').replace('\n', '')
syllables = list(text)
n_syll = len(syllables)

# 特征提取
snd = parselmouth.Sound(signal, sample_rate)
pitch = snd.to_pitch()
intensity = snd.to_intensity()
formant = snd.to_formant_burg()
spectrogram = snd.to_spectrogram(window_length=0.005, maximum_frequency=5000.0)

f0_values = pitch.selected_array['frequency']
intensity_values = intensity.values
times = pitch.xs()
X, Y = np.meshgrid(spectrogram.xs(), spectrogram.ys())
Z = spectrogram.values


print("整段统计特征：")
print(f"F0均值: {np.nanmean(f0_values):.2f} Hz, 标准差: {np.nanstd(f0_values):.2f} Hz")
print(f"强度均值: {np.nanmean(intensity_values):.2f} dB, 标准差: {np.nanstd(intensity_values):.2f} dB")
f1 = [formant.get_value_at_time(1, t) for t in times]
f2 = [formant.get_value_at_time(2, t) for t in times]
print(f"Formant1均值: {np.nanmean(f1):.2f} Hz, Formant2均值: {np.nanmean(f2):.2f} Hz")


spec_energy = Z.sum(axis=0)
spec_energy_db = 10 * np.log10(spec_energy + 1e-8)
threshold = np.percentile(spec_energy_db, 60)
voiced_mask = spec_energy_db > threshold

voiced_segments = []
in_voiced = False
for i, flag in enumerate(voiced_mask):
    t = X[0][i]
    if flag and not in_voiced:
        start = t
        in_voiced = True
    elif not flag and in_voiced:
        end = t
        voiced_segments.append((start, end, end - start))
        in_voiced = False
if in_voiced:
    voiced_segments.append((start, X[0][-1], X[0][-1] - start))

# F0 + Formant
print("有声音且F0和Formant在指定范围内的区间：")
filtered_segments = []
for start, end, duration in voiced_segments:
    f0_seg = f0_values[(times >= start) & (times < end)]
    f0_seg = f0_seg[f0_seg > 0]
    f1_seg = np.array([formant.get_value_at_time(1, t) for t in times[(times >= start) & (times < end)]])
    f2_seg = np.array([formant.get_value_at_time(2, t) for t in times[(times >= start) & (times < end)]])
    f1_seg, f2_seg = f1_seg[~np.isnan(f1_seg)], f2_seg[~np.isnan(f2_seg)]
    if len(f0_seg) == 0 or len(f1_seg) == 0 or len(f2_seg) == 0:
        continue
    f0_median, f0_mean = np.median(f0_seg), np.mean(f0_seg)
    f1_median, f2_median = np.median(f1_seg), np.median(f2_seg)
    if (f0_min <= f0_median <= f0_max or f0_min <= f0_mean <= f0_max) and \
       (F1_MIN <= f1_median <= F1_MAX or F2_MIN <= f2_median <= F2_MAX):
        filtered_segments.append((start, end, duration, f0_median, f0_mean, f1_median, f2_median))

# 合并
def merge_segments(segments, min_gap=MERGE_MIN_GAP):
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg[0] - prev[1] < min_gap:
            new_start = prev[0]
            new_end = seg[1]
            new_duration = new_end - new_start
            new_f0s = [prev[3], seg[3]] if len(prev) > 3 and len(seg) > 3 else []
            new_f0 = np.mean(new_f0s) if new_f0s else np.nan
            merged[-1] = (new_start, new_end, new_duration, new_f0, seg[4], seg[5], seg[6])
        else:
            merged.append(seg)
    return merged

merged_segments = merge_segments(filtered_segments)

print("\n合并后有效区间：")
for i, (start, end, dur, f0_med, f0_mean, f1_med, f2_med) in enumerate(merged_segments):
    print(f"段{i+1}: {start:.2f}s ~ {end:.2f}s, 时长 {dur:.2f}s, F0中位: {f0_med:.2f}, 均值: {f0_mean:.2f}, F1: {f1_med:.1f}, F2: {f2_med:.1f}")

# 非线性音节对齐
syllable2segment = []
if merged_segments and n_syll > 0:
    seg_durations = np.array([end - start for start, end, *_ in merged_segments])
    temp = SYLLABLE_TEMP
    weights = np.exp(seg_durations / (seg_durations.max() * temp))
    weights = weights / weights.sum()
    syllable_counts = np.round(weights * n_syll).astype(int)
    diff = n_syll - syllable_counts.sum()
    for i in range(abs(diff)):
        idx = np.argmax(syllable_counts) if diff < 0 else np.argmin(syllable_counts)
        syllable_counts[idx] += 1 if diff > 0 else -1

    print("\n音节与有声段对齐：")
    print("音节\t起始(s)\t结束(s)")
    cur = 0
    for (start, end, *_), n_this in zip(merged_segments, syllable_counts):
        seg_times = np.linspace(start, end, n_this + 1)
        for i in range(n_this):
            if cur < n_syll:
                s = syllables[cur]
                st, et = seg_times[i], seg_times[i+1]
                syllable2segment.append((s, st, et))
                print(f"{s}\t{st:.2f}\t{et:.2f}")
                cur += 1
else:
    print("无法对齐音节与有声段。")

# 可视化
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
axes[0].plot(times, f0_values, label="F0 (Hz)")
axes[0].plot(intensity.xs(), intensity.values[0], label="Intensity (dB)", alpha=0.7)
axes[0].legend()
axes[0].set_title("F0 and Intensity")
pcm = axes[1].pcolormesh(X, Y, 10 * np.log10(Z), shading='auto', cmap='Greys')
axes[1].set_ylim(0, 5000)
axes[1].set_title("Spectrogram with F0")
fig.colorbar(pcm, ax=axes[1], label='dB')
ax2 = axes[1].twinx()
ax2.plot(times, f0_values, color='b', label='F0 (Hz)', linewidth=2)
ax2.set_ylabel("F0 (Hz)", color='b')
for (start, end, *_ ) in merged_segments:
    axes[1].axvspan(start, end, color='deepskyblue', alpha=0.18)
plt.tight_layout()
plt.savefig("parselmouth_features_subplots2.png", dpi=300)
# plt.show()