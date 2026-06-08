"""
修复GPU_GenderRecognition.py文件中的Unicode字符
"""

import re

# 读取文件
with open(r'c:\Users\27946\Desktop\Sound\Gender\GPU_GenderRecognition.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换 Unicode 字符
replacements = {
    '✓': '[OK]',
    '✗': '[ERROR]',
    '🎯': '[*]'
}

for old, new in replacements.items():
    content = content.replace(old, new)

# 写回文件
with open(r'c:\Users\27946\Desktop\Sound\Gender\GPU_GenderRecognition.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ 文件中的Unicode字符已全部替换")
print("  ✓ -> [OK]")
print("  ✗ -> [ERROR]")
print("  🎯 -> [*]")
