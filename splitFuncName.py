#!/usr/bin/env python3
"""
使用 Clang Python 绑定提取指定目录中所有 isl_* 函数名。

功能：
- 递归遍历目录
- 解析 .c 和 .h 文件
- 跳过模板/测试文件
- 提取 isl_* 函数
- 输出 CSV 文件

依赖：
pip install clang
"""

import os
import sys
import csv
import argparse
from typing import List, Dict, Optional
import clang.cindex
from clang.cindex import Index, CursorKind, TranslationUnit


def get_system_include_paths() -> List[str]:
    """
    使用 'clang -E -v -' 获取系统的标准头文件搜索路径。
    """
    try:
        # 尝试常见的 Clang 路径
        clang_paths = ['clang', 'clang-14', 'clang-15', 'clang-16', 'clang-17']
        cc = os.environ.get('CC', 'clang')  # 尝试环境变量
        clang_paths.insert(0, cc)
        
        for clang_cmd in clang_paths:
            try:
                result = os.popen(f"{clang_cmd} -E -v - 2>&1 < /dev/null").read()
                if "#include <...> search starts here:" in result:
                    lines = result.splitlines()
                    in_search_list = False
                    include_paths = []
                    
                    for line in lines:
                        line = line.strip()
                        if line == '#include <...> search starts here:':
                            in_search_list = True
                            continue
                        if line == 'End of search list.':
                            break
                        if in_search_list and line and not line.startswith('#'):
                            path = line.split()[0]
                            if os.path.isdir(path):
                                include_paths.append(path)
                    return include_paths
            except:
                continue
                
    except Exception as e:
        print(f"⚠️  获取系统头文件路径失败: {e}")
    
    # 失败时返回一个保守的默认列表
    print("⚠️  使用默认头文件路径")
    return [
        '/usr/include',
        '/usr/local/include',
        '/Library/Developer/CommandLineTools/usr/include/c++/v1', # macOS
        '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include', # macOS SDK
    ]


def should_skip_file(filename: str) -> bool:
    """
    判断是否应该跳过此文件。
    """
    skip_keywords = [
        'templ',        # 模板文件
        'test_inputs',  # 测试输入
        'check_',       # 检查文件
        'isl_test',     # 测试文件
        'isl_sample',   # 示例文件
        'isl_benchmark',# 基准测试
        'isl_config',   # 配置
        'isl_config.h', # 配置头文件
        'isl_options',  # 选项
        '__',           # 内部文件
    ]
    filename_lower = filename.lower()
    return any(keyword in filename_lower for keyword in skip_keywords)


def extract_functions_from_file(
    filename: str, 
    include_paths: List[str],
    clang_args: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    使用 Clang 从单个文件中提取函数。
    
    Args:
        filename: C 源文件路径
        include_paths: 头文件搜索路径列表
        clang_args: 额外的 Clang 命令行参数
    
    Returns:
        函数信息列表
    """
    index = Index.create()
    args = []
    
    # 添加头文件路径
    if include_paths:
        args.extend(['-I' + path for path in include_paths])
    
    # 添加额外参数
    if clang_args:
        args.extend(clang_args)
    
    # 强制指定 C 语言 (避免 .h 文件被当作 C++)
    args.append('-x')
    args.append('c')
    
    # 如果文件是 .h，可能需要指定标准
    if filename.endswith('.h'):
        args.append('-std=c11') # 或 c99
    
    functions = []
    
    try:
        # 解析翻译单元
        translation_unit: TranslationUnit = index.parse(
            filename, 
            args=args, 
            options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        )
        
        # 检查诊断信息 (警告和错误)
        for diag in translation_unit.diagnostics:
            severity = diag.severity
            if severity == diag.Warning:
                print(f"  ⚠️  [警告] {diag.location}: {diag.spelling}")
            elif severity >= diag.Error:
                print(f"  ❌ [错误] {diag.location}: {diag.spelling}")
        
        # 遍历 AST 提取函数
        def visit_node(node):
            # 只提取函数定义
            if node.kind == CursorKind.FUNCTION_DECL and node.is_definition():
                func_name = node.spelling
                
                # 过滤：只保留 isl_ 开头的函数
                if func_name.startswith('isl_'):
                    functions.append({
                        'function_name': func_name,
                        'file': os.path.relpath(filename), # 相对路径
                        'line': str(node.location.line),
                        'column': str(node.location.column),
                    })
            
            # 递归访问子节点
            for child in node.get_children():
                visit_node(child)
        
        visit_node(translation_unit.cursor)
        
    except Exception as e:
        print(f"❌ 解析文件失败 {filename}: {type(e).__name__}: {e}")
    
    return functions


def main():
    parser = argparse.ArgumentParser(description="提取 isl 库中的函数名")
    parser.add_argument('source_dir', help='C 源码根目录')
    parser.add_argument('--output', default='isl_functions.csv', help='输出 CSV 文件名')
    parser.add_argument('--include-ext', nargs='+', default=['.c', '.h'], 
                       help='要处理的文件扩展名 (默认: .c .h)')
    args = parser.parse_args()
    
    source_dir = args.source_dir
    output_file = args.output
    valid_extensions = set(args.include_ext)
    
    # 检查源码目录
    if not os.path.isdir(source_dir):
        print(f"❌ 错误: 目录不存在 '{source_dir}'")
        sys.exit(1)
    
    # 获取系统头文件路径
    print("🔍 正在检测系统头文件路径...")
    include_paths = get_system_include_paths()
    print(f"✅ 找到 {len(include_paths)} 个系统头文件路径")
    # 如果需要，可以打印前几个
    # for p in include_paths[:3]:
    #     print(f"   {p}")
    
    # 额外的 Clang 参数 (通常不需要)
    clang_args = []
    # 示例: clang_args = ['-DDEBUG=0', '-DISL_DEBUG=0']
    
    # 收集所有函数
    all_functions = []
    
    print(f"📁 正在遍历目录: {source_dir}")
    file_count = 0
    
    # 递归遍历目录
    for root, dirs, files in os.walk(source_dir):
        # 跳过隐藏目录 (如 .git, __pycache__)
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            filepath = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            
            # 检查扩展名
            if ext.lower() not in valid_extensions:
                continue
                
            # 检查是否跳过
            if should_skip_file(file):
                # print(f"⏭️  跳过: {filepath}")
                continue
            
            file_count += 1
            print(f"🔍 正在解析: {filepath}")
            
            funcs = extract_functions_from_file(filepath, include_paths, clang_args)
            all_functions.extend(funcs)
            
            if not funcs:
                print(f"   ⚠️  未找到 isl_* 函数")
            else:
                for func in funcs:
                    print(f"   📌 {func['function_name']} (行 {func['line']})")
    
    print(f"\n✅ 完成！共处理 {file_count} 个文件，找到 {len(all_functions)} 个 isl_* 函数。")
    
    # 保存到 CSV
    if all_functions:
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['function_name', 'file', 'line', 'column']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for func in all_functions:
                    writer.writerow(func)
            print(f"💾 函数列表已保存至: {output_file}")
        except Exception as e:
            print(f"❌ 保存 CSV 文件失败: {e}")
            sys.exit(1)
    else:
        print("❌ 未找到任何 isl_* 函数。")


if __name__ == '__main__':
    # --- 重要：设置 libclang 库路径 ---
    # 如果报错 "libclang.so not found"，请取消注释并设置正确路径
    # 
    # Linux (Ubuntu): 
    # clang.cindex.Config.set_library_file('/usr/lib/llvm-14/lib/libclang.so.1')
    # 
    # macOS (Homebrew): 
    # clang.cindex.Config.set_library_file('/opt/homebrew/lib/libclang.dylib')
    # 或
    # clang.cindex.Config.set_library_path('/opt/homebrew/lib')
    #
    # Windows:
    # clang.cindex.Config.set_library_file('C:\\Program Files\\LLVM\\bin\\libclang.dll')
    
    # 尝试自动查找 (通常不需要)
    try:
        # 这会尝试加载系统默认的 libclang
        Index.create()
    except Exception as e:
        print(f"❌ 无法加载 libclang。请检查 Clang 是否安装，并可能需要手动设置路径。")
        print(f"   错误: {e}")
        print("   提示: 取消注释脚本末尾的 Config.set_library_file() 并设置正确路径。")
        sys.exit(1)
    
    main()
