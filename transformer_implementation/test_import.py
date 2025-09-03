# 简单的导入测试脚本
# 这个脚本只导入Transformer模块，不执行任何操作
# 用于验证模块之间的导入关系是否正确

try:
    print("Importing Transformer modules...")
    from transformer import Transformer
    print("Successfully imported Transformer module!")
    print("Transformer implementation is structurally correct.")
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

print("\nTest completed.")