
def increment(a):
    # 函数内执行 a = a + 1
    a = a + 1
    return a

# 初始化变量 a
a = 0

# 循环调用函数 10 次
for _ in range(10):
    a = increment(a)
    print(f"Current value of a: {a}")
path = '_/5e-6/ds_swin_base_patch4_window7_224/2024_09_1'
# 计算字符串长度
path_length = len(path)
print(f"The length of the path string is: {path_length}")



        # 定义列表
numbers = [1.3689e-06, 1.4325e-06, 2.2655e-07, 5.4692e-06, 2.7261e-06, 2.8706e-06,
        4.4553e-03, 2.0927e-06, 3.9871e-07, 8.2118e-07, 9.9553e-01, 1.4405e-07]

# 计算和
total_sum = sum(numbers)
print(total_sum)
