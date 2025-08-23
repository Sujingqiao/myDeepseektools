import os
import re
import argparse

def extract_long_comments_from_file(file_path, min_lines=5):
    """
    从单个C源码文件中提取超过指定行数的注释块
    
    Args:
        file_path (str): C源码文件路径
        min_lines (int): 注释块的最小行数阈值
    
    Returns:
        list: 包含注释块和文件信息的字典列表
    """
    long_comments = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            
            # 使用正则表达式匹配多行注释块
            pattern = r'/\*.*?\*/'
            comments = re.findall(pattern, content, re.DOTALL)
            
            for comment in comments:
                # 计算注释行数
                lines = comment.split('\n')
                line_count = len([line for line in lines if line.strip()])
                
                if line_count >= min_lines:
                    # 添加文件信息和注释内容
                    long_comments.append({
                        'file_path': file_path,
                        'line_count': line_count,
                        'content': comment
                    })
                    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    return long_comments

def extract_comments_from_directory(directory, output_file, min_lines=5, extensions=None):
    """
    遍历目录中的所有C源码文件，提取长注释并保存到输出文件
    
    Args:
        directory (str): 要遍历的目录路径
        output_file (str): 输出文件路径
        min_lines (int): 注释块的最小行数阈值
        extensions (list): 要处理的文件扩展名列表
    """
    if extensions is None:
        extensions = ['.c', '.h', '.cpp', '.hpp', '.cc', '.cxx']
    
    all_long_comments = []
    
    # 遍历目录树
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                print(f"正在处理: {file_path}")
                
                comments = extract_long_comments_from_file(file_path, min_lines)
                all_long_comments.extend(comments)
    
    # 将结果写入输出文件
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for comment_info in all_long_comments:
            out_file.write(f"文件路径: {comment_info['file_path']}\n")
            out_file.write(f"注释行数: {comment_info['line_count']}\n")
            out_file.write("注释内容:\n")
            out_file.write(comment_info['content'])
            out_file.write("\n" + "="*80 + "\n\n")
    
    print(f"完成! 共找到 {len(all_long_comments)} 个超过 {min_lines} 行的注释块")
    print(f"结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='提取C源码中超过指定行数的注释块')
    parser.add_argument('directory', help='要遍历的目录路径')
    parser.add_argument('-o', '--output', default='long_comments.txt', 
                       help='输出文件路径 (默认: long_comments.txt)')
    parser.add_argument('-l', '--lines', type=int, default=5,
                       help='注释块的最小行数阈值 (默认: 5)')
    parser.add_argument('-e', '--extensions', nargs='+', 
                       default=['.c', '.h', '.cpp', '.hpp', '.cc', '.cxx'],
                       help='要处理的文件扩展名 (默认: .c .h .cpp .hpp .cc .cxx)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"错误: 目录 '{args.directory}' 不存在")
        return
    
    extract_comments_from_directory(
        args.directory, 
        args.output, 
        args.lines, 
        args.extensions
    )

if __name__ == "__main__":
    main()
