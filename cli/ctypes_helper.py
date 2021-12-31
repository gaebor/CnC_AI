import re
import sys
import ctypes


def main():
    struct_pattern = re.compile(r'\s*struct\s+(\w+)\s*{([^}]*)}\s*;')
    members_pattern = re.compile(r'\s*(\w+(\s+\w+)*)\s+(\w+)(\[(\w+)\])?\s*;\s*(//[^\n]*)?')
    content = sys.stdin.read()
    for struct in struct_pattern.finditer(content):
        struct_name, struct_content = struct.groups()
        print(f'class {struct_name}(CncStruct):')
        print('    _fields_ = [')
        for member in members_pattern.finditer(struct_content):
            member_type, _, member_name, _, array_size, _ = member.groups()
            if hasattr(ctypes, f'c_{member_type}'):
                member_type = f'ctypes.c_{member_type}'
            else:
                member_type = member_type

            if array_size is not None:
                member_type += ' * ' + array_size
            print(f'        (\'{member_name}\', {member_type}),')
        print('    ]')


if __name__ == '__main__':
    main()
