from textgrid import TextGrid
import sys
import os

def print_textgrid_info(textgrid_path):
    tg = TextGrid.fromFile(textgrid_path)

    for tier in tg.tiers:
        print(f"Tier: {tier.name}")
        for interval in tier.intervals:
            if interval.mark.strip():  # 只打印非空标签
                print(f"{interval.mark} [{interval.minTime:.2f} - {interval.maxTime:.2f}]")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python view_alignment.py path/to/TextGrid")
        sys.exit(1)

    textgrid_file = sys.argv[1]
    if not os.path.isfile(textgrid_file):
        print("错误：TextGrid 文件不存在")
        sys.exit(1)

    print_textgrid_info(textgrid_file)