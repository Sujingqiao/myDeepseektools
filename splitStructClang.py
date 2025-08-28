# extract_structs_final_fixed.py
import os
import sys
from clang.cindex import Index, CursorKind

def is_preprocessor_line(line):
    """判断是否为预处理指令"""
    stripped = line.strip()
    return stripped.startswith('#')

def get_struct_source_clean(cursor, lines):
    """提取 struct 主体，跳过 #include/#define"""
    try:
        start_line = max(0, cursor.extent.start.line - 1)
        end_line = min(len(lines), cursor.extent.end.line)

        content = []
        brace_count = 0
        in_body = False

        for i in range(start_line, end_line):
            line = lines[i].rstrip('\n\r')
            if '{' in line:
                brace_count += line.count('{')
                in_body = True
            if in_body and not is_preprocessor_line(line):
                content.append(line)
            if '}' in line:
                brace_count -= line.count('}')
                if brace_count <= 0:
                    break
        return content
    except:
        return []

def is_top_level_struct(cursor):
    """判断是否在顶层（不在函数、嵌套结构中）"""
    parent = cursor.semantic_parent
    if parent is None:
        return True
    # 排除在函数内部
    if parent.kind in [CursorKind.FUNCTION_DECL]:
        return False
    # 排除在 C++ 方法中（虽然你是 C，但安全起见）
    if parent.kind in [CursorKind.CXX_METHOD]:
        return False
    # 排除在匿名结构/联合内部
    if parent.kind in [CursorKind.STRUCT_DECL, CursorKind.UNION_DECL] and not parent.spelling:
        return False
    return True

def is_valid_struct_definition(cursor):
    """是否为有效的顶层 struct 定义"""
    if cursor.kind != CursorKind.STRUCT_DECL:
        return False
    if not cursor.location.file:
        return False
    if cursor.is_anonymous():
        return False
    if not is_top_level_struct(cursor):
        return False
    # 必须有 body（多行）
    if cursor.extent.start.line == cursor.extent.end.line:
        return False
    return True

def traverse_ast(node, lines, structs, file_name, seen):
    """递归遍历 AST"""
    if is_valid_struct_definition(node):
        struct_name = node.spelling
        file_key = file_name
        line_key = node.extent.start.line
        name_key = f"{file_key}:{struct_name}"

        if name_key in seen or (file_key, line_key) in seen:
            return

        content = get_struct_source_clean(node, lines)
        if content:
            structs.append({
                'file': file_name,
                'line': node.extent.start.line,
                'body': content
            })
            seen.add(name_key)
            seen.add((file_key, line_key))

    # 递归子节点
    for child in node.get_children():
        traverse_ast(child, lines, structs, file_name, seen)

def parse_file(file_path):
    """解析单个文件"""
    index = Index.create()
    try:
        # 使用 c99 标准解析 C 代码
        tu = index.parse(file_path, args=['-x', 'c', '-std=c99'])
        if not tu:
            print(f"❌ 无法解析: {file_path}")
            return []
    except Exception as e:
        print(f"❌ 解析失败 {file_path}: {e}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.rstrip('\n\r') for line in f.readlines()]
    except Exception as e:
        print(f"❌ 读取失败 {file_path}: {e}")
        return []

    filename = os.path.basename(file_path)
    structs = []
    seen = set()
    traverse_ast(tu.cursor, lines, structs, filename, seen)
    return structs

def main(directory):
    if not os.path.isdir(directory):
        print(f"❌ 错误: 目录不存在: {directory}")
        return

    all_structs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.c', '.h')):
                path = os.path.join(root, file)
                print(f"🔍 {path}")
                all_structs.extend(parse_file(path))

    # 排序输出
    all_structs.sort(key=lambda x: (x['file'], x['line']))

    # 写入文件
    with open('structs_final.txt', 'w', encoding='utf-8') as f:
        for s in all_structs:
            title = f"{s['file']}:{s['line']}"
            padding = (80 - len(title)) // 2
            left = '$' * padding
            right = '$' * (80 - len(title) - padding)
            f.write(left + title + right + '\n')
            for line in s['body']:
                f.write(line + '\n')
            f.write('\n')  # 空行分隔

    print(f"✅ 成功提取 {len(all_structs)} 个 struct 定义")
    print(f"📄 输出文件: structs_final.txt")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("📌 用法: python extract_structs_final_fixed.py <C代码目录>")
    else:
        main(sys.argv[1])
