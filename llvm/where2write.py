from clang.cindex import *

# 我们要监控的结构体名和字段
STRUCT_NAME = 'isl_map'
FIELDS = {
    'ref', 'flags', 'cached_simple_hull', 'ctx', 'dim', 'n', 'size', 'p'
}
FLAGS = {
    'ISL_MAP_DISJOINT', 'ISL_MAP_NORMALIZED',
    'ISL_SET_DISJOINT', 'ISL_SET_NORMALIZED'
}

def is_isl_map_type(type):
    """判断类型是否为 struct isl_map"""
    decl = type.get_declaration()
    return (decl.kind == CursorKind.STRUCT_DECL and
            decl.spelling == STRUCT_NAME)

def is_field_access(cursor, target_field):
    """判断是否访问了 map->field"""
    if cursor.kind == CursorKind.MEMBER_REF:
        member_name = cursor.spelling
        base = cursor.get_definition()
        if base and member_name == target_field:
            # 检查 base 的类型是否为 isl_map*
            base_type = base.type
            if base_type.kind == TypeKind.POINTER:
                pointee = base_type.get_pointee()
                if is_isl_map_type(pointee):
                    return True
    return False

def is_flag_constant(token):
    """判断 token 是否为 ISL_* 标志常量"""
    return token.spelling in FLAGS

def find_map_modifications(filename, args=[]):
    index = Index.create()
    tu = index.parse(filename, args=args)

    modifications = []

    def walk(node):
        # 1. 检查是否为赋值、复合赋值、自增等写操作
        if node.kind in (
            CursorKind.BINARY_OPERATOR,      # +=, |=, &=
            CursorKind.COMPOUND_ASSIGNMENT_OPERATOR,
            CursorKind.UNARY_OPERATOR,       # ++, --
        ):
            # 获取操作符的 token
            op_token = list(node.get_tokens())[1]  # 通常是操作符
            op_str = op_token.spelling

            # 只关心写操作
            if op_str in ('=', '+=', '-=', '|=', '&=', '++', '--'):
                # 检查左操作数是否是 isl_map 字段
                children = list(node.get_children())
                if children:
                    lhs = children[0]
                    for field in FIELDS:
                        if is_field_access(lhs, field):
                            modifications.append({
                                'file': filename,
                                'line': node.location.line,
                                'column': node.location.column,
                                'field': field,
                                'op': op_str,
                                'code': ''.join(t.spelling for t in node.get_tokens())
                            })
                            break

        # 2. 检查函数调用是否可能修改 map
        elif node.kind == CursorKind.CALL_EXPR:
            callee = node.get_called_function()
            if callee:
                func_name = callee.spelling
                # 常见的 setter 函数
                if func_name.startswith('isl_map_') and 'set' in func_name.lower():
                    # 检查第一个参数是否是 isl_map*
                    args = list(node.get_arguments())
                    if args:
                        arg0 = args[0]
                        if arg0.type.kind == TypeKind.POINTER:
                            pointee = arg0.type.get_pointee()
                            if is_isl_map_type(pointee):
                                modifications.append({
                                    'file': filename,
                                    'line': node.location.line,
                                    'column': node.location.column,
                                    'field': 'unknown_via_function',
                                    'op': 'call',
                                    'func': func_name,
                                    'code': ''.join(t.spelling for t in node.get_tokens())
                                })

        # 3. 检查 memset/memcpy 等内存操作
        elif node.kind == CursorKind.CALL_EXPR:
            callee = node.get_called_function()
            if callee and callee.spelling in ('memset', 'memcpy', 'bzero'):
                args = list(node.get_arguments())
                if args:
                    dest = args[0]
                    if dest.type.kind == TypeKind.POINTER:
                        pointee = dest.type.get_pointee()
                        if is_isl_map_type(pointee):
                            modifications.append({
                                'file': filename,
                                'line': node.location.line,
                                'column': node.location.column,
                                'field': 'entire_struct',
                                'op': 'memory_op',
                                'func': callee.spelling,
                                'code': ''.join(t.spelling for t in node.get_tokens())
                            })

        # 递归子节点
        for child in node.get_children():
            walk(child)

    walk(tu.cursor)
    return modifications
