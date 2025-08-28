# extract_structs_final_clean.py
import os
import re

# 匹配 struct 开头：struct name { 或 struct {（匿名）
STRUCT_START_PATTERN = re.compile(
    r'\b(?:typedef\s+|static\s+|extern\s+|const\s+)*'
    r'struct\s+(\w*)\s*'  # 捕获名字
    r'(?={)',
    re.IGNORECASE
)

# 用于判断是否是初始化（排除 struct var = { ... };）
INIT_PATTERN = re.compile(r'=\s*{')

# 全局去重集合
seen_structs = set()

def extract_struct_body_lines(lines, start_idx):
    """提取 struct 主体，跳过初始化"""
    brace_count = 0
    struct_lines = []
    i = start_idx

    # 找到第一个 { 开始
    while i < len(lines) and '{' not in lines[i]:
        i += 1
    if i >= len(lines):
        return []

    while i < len(lines):
        line = lines[i].rstrip('\n\r')
        struct_lines.append(line)

        brace_count += line.count('{')
        brace_count -= line.count('}')

        if brace_count <= 0 and line.strip().endswith(';'):
            break
        i += 1

    # 检查是否是初始化
    full_text = ' '.join(struct_lines)
    if INIT_PATTERN.search(full_text):
        return []

    return struct_lines

def is_in_struct_body(lines, start_idx):
    """检查当前行是否在某个 struct 主体内（避免嵌套 struct）"""
    brace_count = 0
    # 从文件开头到 start_idx，看是否有未闭合的 struct {
    for i in range(start_idx):
        line = lines[i]
        # 忽略注释
        line = re.sub(r'//.*|/\*.*?\*/', '', line)
        if 'struct' in line and '{' in line:
            brace_count += line.count('{')
        elif '}' in line:
            brace_count -= line.count('}')
    return brace_count > 0

def get_struct_name_from_line(line):
    """从 struct 行提取名字，无名返回 None"""
    match = STRUCT_START_PATTERN.search(line)
    if match:
        return match.group(1).strip() or None
    return None

def find_structs_in_file(file_path):
    structs = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.rstrip('\n\r') for line in f.readlines()]

        filename = os.path.basename(file_path)

        for line_idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('/*') or 'struct' not in stripped:
                continue

            if STRUCT_START_PATTERN.search(line):
                # 跳过 struct 内部定义的 struct
                if is_in_struct_body(lines, line_idx):
                    continue

                body = extract_struct_body_lines(lines, line_idx)
                if not body:
                    continue

                # 提取 struct 名字
                struct_name = get_struct_name_from_line(line)
                if not struct_name:
                    # 如果是匿名 struct + typedef，尝试从 typedef 找名字
                    # typedef struct { ... } foo;
                    typedef_match = re.search(r'typedef\s+struct\s+{', line)
                    if not typedef_match:
                        continue
                    # 扫描这一行或下一行找名字
                    full_line = ' '.join(body).strip()
                    name_match = re.search(r'typedef\s+struct\s+{[^}]*};\s*(\w+);', full_line)
                    if name_match:
                        struct_name = name_match.group(1)

                # 如果还是没有名字，跳过（匿名且无 typedef）
                if not struct_name:
                    continue

                # 去重 key: (struct_name, file_keyword)
                file_keyword = filename.split('.')[0]  # 用文件名前缀作为关键词
                key = (struct_name, file_keyword)

                if key in seen_structs:
                    continue

                seen_structs.add(key)
                structs.append({
                    'file': filename,
                    'line': line_idx + 1,
                    'name': struct_name,
                    'body': body
                })

    except Exception as e:
        print(f"Error: {file_path}: {e}")
    return structs

def scan_and_save(root_dir, output_file='structs_final_clean.txt'):
    all_structs = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.c', '.h')):
                file_path = os.path.join(dirpath, filename)
                all_structs.extend(find_structs_in_file(file_path))

    # 排序：按文件、行号
    all_structs.sort(key=lambda x: (x['file'], x['line']))

    with open(output_file, 'w', encoding='utf-8') as f:
        for s in all_structs:
            title = f"{s['file']}:{s['line']} [{s['name']}]"
            pad = (80 - len(title)) // 2
            left = '$' * pad
            right = '$' * (80 - len(title) - pad)
            f.write(left + title + right + '\n')
            for line in s['body']:
                f.write(line + '\n')
            f.write('\n')

    print(f"✅ 终极清理版完成 → {output_file}")
    print(f"📊 共提取 {len(all_structs)} 个 struct 定义")
    print(f"🔍 去重关键词: (结构体名, 文件名前缀)")

def main(directory):
    if not os.path.isdir(directory):
        print(f"❌ 错误: 目录不存在: {directory}")
        return
    scan_and_save(directory)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("📌 用法: python extract_structs_final_clean.py <C代码目录>")
    else:
        main(sys.argv[1])
