import os
import re
import argparse

def contains_math_formula(comment):
    """
    检查注释是否包含数学公式推导
    
    Args:
        comment (str): 注释内容
        
    Returns:
        bool: 如果包含数学公式推导返回True，否则返回False
    """
    # 匹配常见的数学推导模式
    patterns = [
        r'\[.*?->.*?\]',      # [A -> B] 模式
        r'\(.*?→.*?\)',       # (A → B) 模式
        r'[A-Z]\s*->\s*[A-Z]',  # A -> B 模式
        r'[A-Z]\s*→\s*[A-Z]',   # A → B 模式
        r'[A-Z]\s*=\s*[A-Z]',   # A = B 模式
        r'[A-Z]\s*:\s*[A-Z]',   # A : B 模式
        r'\\[a-zA-Z]+\{.*?\}',  # LaTeX 公式
        r'f\(.*?\)\s*=',        # 函数定义
        r'∀|∃|∈|⊆|⊂|∪|∩|¬|∧|∨|⇒|⇔',  # 数学符号
        r'[Ss]et.*->.*',        # Set A -> B 模式
        r'map(ping)?.*->.*',    # mapping A -> B 模式
    ]
    
    for pattern in patterns:
        if re.search(pattern, comment, re.IGNORECASE):
            return True
    
    return False

def contains_ascii_diagram(comment):
    """
    检查注释是否包含ASCII图
    
    Args:
        comment (str): 注释内容
        
    Returns:
        bool: 如果包含ASCII图返回True，否则返回False
    """
    # 检查ASCII图的特征
    lines = comment.split('\n')
    
    # 检查是否有连续多行包含绘图字符
    drawing_chars = r'[\-+|/\*_=#@]'
    drawing_lines = 0
    
    for line in lines:
        # 跳过空行和只有星号的行（可能是注释边框）
        stripped = line.strip()
        if not stripped or stripped == '*' or stripped.startswith('* '):
            continue
            
        # 计算绘图字符的比例
        if len(stripped) > 0:
            drawing_char_count = len(re.findall(drawing_chars, stripped))
            if drawing_char_count / len(stripped) > 0.3:  # 30%以上的字符是绘图字符
                drawing_lines += 1
    
    # 如果有至少3行看起来像绘图，则认为包含ASCII图
    return drawing_lines >= 3

def is_pointer_only_comment(comment):
    """
    检查注释是否只是简单的指针引用或变量描述
    
    Args:
        comment (str): 注释内容
        
    Returns:
        bool: 如果只是简单指针引用返回True，否则返回False
    """
    # 过滤掉只包含变量名、指针引用或简单描述的注释
    patterns = [
        r'^\s*/\*\s*[A-Za-z_][A-Za-z0-9_]*(\s*->\s*[A-Za-z_][A-Za-z0-9_]*)*\s*\*/\s*$',  # 单行指针注释
        r'Set\s+[A-Za-z_][A-Za-z0-9_]*\s*->\s*[A-Za-z_][A-Za-z0-9_]*',  # Set var->var
        r'Update\s+[A-Za-z_][A-Za-z0-9_]*\s*->\s*[A-Za-z_][A-Za-z0-9_]*',  # Update var->var
        r'Return\s+[A-Za-z_][A-Za-z0-9_]*',  # Return var
        r'Initialize\s+[A-Za-z_][A-Za-z0-9_]*',  # Initialize var
    ]
    
    for pattern in patterns:
        if re.search(pattern, comment, re.IGNORECASE):
            return True
    
    # 检查注释是否只包含简单的变量描述
    lines = comment.split('\n')
    simple_lines = 0
    for line in lines:
        stripped = line.strip()
        # 跳过空行和注释标记
        if not stripped or stripped in ['/*', '*/'] or stripped.startswith('* '):
            continue
            
        # 检查是否只是简单的变量描述
        if re.match(r'^\s*\*?\s*[A-Za-z_][A-Za-z0-9_]*(\s*->\s*[A-Za-z_][A-Za-z0-9_]*)*\s*$', stripped):
            simple_lines += 1
    
    # 如果所有非空行都是简单变量描述，则过滤掉
    if simple_lines > 0 and simple_lines == len([l for l in lines if l.strip() and not l.strip() in ['/*', '*/']]):
        return True
    
    return False

def extract_special_comments_from_file(file_path):
    """
    从单个C源码文件中提取包含数学公式或ASCII图的注释块
    
    Args:
        file_path (str): C源码文件路径
        
    Returns:
        list: 包含特殊注释块和文件信息的字典列表
    """
    special_comments = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            
            # 使用正则表达式匹配多行注释块
            pattern = r'/\*.*?\*/'
            comments = re.findall(pattern, content, re.DOTALL)
            
            for comment in comments:
                # 过滤掉简单的指针注释
                if is_pointer_only_comment(comment):
                    continue
                    
                # 检查是否包含数学公式或ASCII图
                if contains_math_formula(comment) or contains_ascii_diagram(comment):
                    # 计算注释行数
                    lines = comment.split('\n')
                    line_count = len([line for line in lines if line.strip()])
                    
                    # 添加文件信息和注释内容
                    special_comments.append({
                        'file_path': file_path,
                        'line_count': line_count,
                        'content': comment
                    })
                    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    return special_comments

def extract_comments_from_directory(directory, output_file, extensions=None):
    """
    遍历目录中的所有C源码文件，提取特殊注释并保存到输出文件
    
    Args:
        directory (str): 要遍历的目录路径
        output_file (str): 输出文件路径
        extensions (list): 要处理的文件扩展名列表
    """
    if extensions is None:
        extensions = ['.c', '.h', '.cpp', '.hpp', '.cc', '.cxx']
    
    all_special_comments = []
    
    # 遍历目录树
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                print(f"正在处理: {file_path}")
                
                comments = extract_special_comments_from_file(file_path)
                all_special_comments.extend(comments)
    
    # 将结果写入输出文件
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for comment_info in all_special_comments:
            out_file.write(f"文件路径: {comment_info['file_path']}\n")
            out_file.write(f"注释行数: {comment_info['line_count']}\n")
            out_file.write("注释内容:\n")
            out_file.write(comment_info['content'])
            out_file.write("\n" + "="*80 + "\n\n")
    
    print(f"完成! 共找到 {len(all_special_comments)} 个包含数学公式或ASCII图的注释块")
    print(f"结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='提取C源码中包含数学公式推导或ASCII图的注释块')
    parser.add_argument('directory', help='要遍历的目录路径')
    parser.add_argument('-o', '--output', default='special_comments.txt', 
                       help='输出文件路径 (默认: special_comments.txt)')
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
        args.extensions
    )

if __name__ == "__main__":
    main()
