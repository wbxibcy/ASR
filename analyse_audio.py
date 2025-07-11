import os
import numpy as np
import soundfile as sf
import parselmouth

def analyze_audio(
    wav_path,
    txt_path,
    sex='male',
    f0_range=None,
    f1_range=(200, 1000),
    f2_range=(600, 3000),
    merge_min_gap=0.05,
    syllable_temp=0.8,
    textgrid_dir='textgrid_out'
):
    # F0范围
    default_f0_range = {
        'male':   (80, 190),
        'female': (165, 255),
        'child':  (250, 400),
        'other':  (50, 500)
    }
    if f0_range is None:
        f0_min, f0_max = default_f0_range.get(sex, default_f0_range['other'])
    else:
        f0_min, f0_max = f0_range

    F1_MIN, F1_MAX = f1_range
    F2_MIN, F2_MAX = f2_range

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
    formant = snd.to_formant_burg()
    spectrogram = snd.to_spectrogram(window_length=0.005, maximum_frequency=5000.0)

    f0_values = pitch.selected_array['frequency']
    times = pitch.xs()
    X, Y = np.meshgrid(spectrogram.xs(), spectrogram.ys())
    Z = spectrogram.values

    # 有声音帧检测
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

    # F0 + Formant筛选
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
    def merge_segments(segments, min_gap=merge_min_gap):
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

    # 非线性音节对齐
    syllable2segment = []
    if merged_segments and n_syll > 0:
        seg_durations = np.array([end - start for start, end, *_ in merged_segments])
        temp = syllable_temp
        weights = np.exp(seg_durations / (seg_durations.max() * temp))
        weights = weights / weights.sum()
        syllable_counts = np.round(weights * n_syll).astype(int)
        diff = n_syll - syllable_counts.sum()
        for i in range(abs(diff)):
            idx = np.argmax(syllable_counts) if diff < 0 else np.argmin(syllable_counts)
            syllable_counts[idx] += 1 if diff > 0 else -1

        cur = 0
        for (start, end, *_), n_this in zip(merged_segments, syllable_counts):
            seg_times = np.linspace(start, end, n_this + 1)
            for i in range(n_this):
                if cur < n_syll:
                    s = syllables[cur]
                    st, et = seg_times[i], seg_times[i+1]
                    syllable2segment.append((s, st, et))
                    cur += 1

    # 输出为TextGrid
    if syllable2segment:
        os.makedirs(textgrid_dir, exist_ok=True)
        # 文件名与音频/文本一致
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        textgrid_path = os.path.join(textgrid_dir, f"{base_name}.TextGrid")
        xmin = 0.0
        xmax = float(times[-1])
        with open(textgrid_path, 'w', encoding='utf-8') as fout:
            fout.write('File type = "ooTextFile"\n')
            fout.write('Object class = "TextGrid"\n\n')
            fout.write(f'xmin = {xmin}\n')
            fout.write(f'xmax = {xmax}\n')
            fout.write('tiers? <exists>\n')
            fout.write('size = 1\n')
            fout.write('item []:\n')
            fout.write('    item [1]:\n')
            fout.write('        class = "IntervalTier"\n')
            fout.write('        name = "syllable"\n')
            fout.write(f'        xmin = {xmin}\n')
            fout.write(f'        xmax = {xmax}\n')
            fout.write(f'        intervals: size = {len(syllable2segment)}\n')
            for idx, (label, st, et) in enumerate(syllable2segment, 1):
                fout.write(f'        intervals [{idx}]:\n')
                fout.write(f'            xmin = {st:.6f}\n')
                fout.write(f'            xmax = {et:.6f}\n')
                fout.write(f'            text = "{label}"\n')
        print(f"TextGrid saved to {textgrid_path}")
    else:
        print("无有效音节对齐结果，未生成TextGrid。")

    return {
        "merged_segments": merged_segments,
        "syllable2segment": syllable2segment
    }

# 示例调用
if __name__ == "__main__":
    analyze_audio(
        wav_path='data/1000185_306eba96.wav',
        txt_path='data/1000185_306eba96.text',
        sex='male',
        textgrid_dir='textgrid_out'
    )