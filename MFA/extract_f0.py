import sys
import pyworld as pw
import numpy as np
import librosa
import tgt
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def extract_f0_and_words(wav_path, textgrid_path):
    # 加载音频
    x, fs = librosa.load(wav_path, sr=None)
    if x.dtype != np.float64:
        x = x.astype(np.float64)

    # 提取 F0
    f0, timeaxis = pw.harvest(x, fs)
    f0 = pw.stonemask(x, f0, timeaxis, fs)

    # 读取对齐结果
    tg = tgt.io.read_textgrid(textgrid_path)
    word_tier = tg.get_tier_by_name("words")

    word_intervals = []
    word_f0s = []
    for interval in word_tier.intervals:
        word = interval.text.strip()
        if not word:
            continue
        start_time = interval.start_time
        end_time = interval.end_time

        indices = np.where((timeaxis >= start_time) & (timeaxis <= end_time))[0]
        f0_values = f0[indices]
        f0_values = f0_values[f0_values > 0]
        if len(f0_values) > 0:
            avg_f0 = np.mean(f0_values)
        else:
            avg_f0 = 0
        
        word_intervals.append((word, start_time, end_time))
        word_f0s.append((word, avg_f0))

    return f0, timeaxis, word_f0s, word_intervals

def plot_f0s_and_annotations(f0, timeaxis, word_f0s, word_intervals):
    fig, axs = plt.subplots(2, 1, figsize=(16, 9))

    # 子图1：完整的 F0 曲线图
    axs[0].plot(timeaxis, f0, color='purple')
    axs[0].set_title("完整的 F0 曲线", fontsize=14)
    axs[0].set_xlabel("时间 (s)")
    axs[0].set_ylabel("F0 (Hz)")
    # axs[0].grid(True)
    # axs[1].xaxis.set_major_locator(ticker.MultipleLocator(0.1))

    # 子图2：完整的 F0 曲线 + 词时间戳和文本注释
    axs[1].plot(timeaxis, f0, color='purple')
    axs[1].set_title("F0曲线与词级时间轴标签", fontsize=14)
    axs[1].set_xlabel("时间 (s)")
    axs[1].set_ylabel("F0 (Hz)")
    axs[1].grid(True)
    axs[1].xaxis.set_major_locator(ticker.MultipleLocator(0.1))

    # 统计每个时间点是开始、结束还是两者皆是
    time_types = defaultdict(set)
    for _, start_time, end_time in word_intervals:
        time_types[start_time].add('start')
        time_types[end_time].add('end')

    # 绘制垂直线，根据时间点类型设置颜色
    for t, types in time_types.items():
        if types == {'start'}:
            color = 'green'   # 仅开始
        elif types == {'end'}:
            color = 'red'     # 仅结束
        else:
            color = 'blue'    # 同时是开始和结束

        axs[1].axvline(x=t, color=color, linestyle='--', alpha=0.8)

    plt.tight_layout()
    plt.savefig("f0_and_word_annotations.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python extract_f0.py path/to/audio.wav path/to/TextGrid")
        sys.exit(1)
    
    wav_path = sys.argv[1]
    textgrid_path = sys.argv[2]

    f0, timeaxis, word_f0s, word_intervals = extract_f0_and_words(wav_path, textgrid_path)

    print("每个词的平均 F0：")
    for word, f in word_f0s:
        print(f"{word}: {f:.2f} Hz")
    print("词划分时间:")
    for word, start, end in word_intervals:
        print(f"  {word:<4}  起始: {start:>5.2f}s  结束: {end:>5.2f}s  时长: {end - start:>5.2f}s")

    plot_f0s_and_annotations(f0, timeaxis, word_f0s, word_intervals)