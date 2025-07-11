import os

# 输入文件路径
input_path = 'data/488480790-1-192.text'
# 输出文件夹
output_dir = 'data/xiaoshan'
os.makedirs(output_dir, exist_ok=True)

with open(input_path, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

for idx, word in enumerate(lines, 1):
    out_path = os.path.join(output_dir, f'{idx:03d}.text')
    with open(out_path, 'w', encoding='utf-8') as fout:
        fout.write(word)
