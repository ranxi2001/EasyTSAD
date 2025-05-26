pip install EasyTSAD
pip install statsmodels
cd Examples/run_your_algo
pip install -r requirements.txt

`BaseMethod` 作为 `MTSExample` 的父类（通过继承）是因为 Python 支持**类继承**，这是一种面向对象编程的核心特性。通过继承，`MTSExample` 可以复用 `BaseMethod` 中的属性和方法，同时可以扩展或重写它的功能。

### 继承的作用
1. **代码复用**：`MTSExample` 可以直接使用 `BaseMethod` 中定义的功能，而无需重新实现。
2. **扩展功能**：`MTSExample` 可以在 `BaseMethod` 的基础上添加新的方法或属性。
3. **多态性**：通过继承，`MTSExample` 可以被视为 `BaseMethod` 的一种类型，方便在框架中统一处理。

### 为什么 `BaseMethod` 是输入
在你的代码中，`MTSExample` 继承了 `BaseMethod`，这意味着：
- `MTSExample` 是 `BaseMethod` 的子类。
- `MTSExample` 可以使用 `BaseMethod` 中的接口（方法和属性）。
- 如果 `EasyTSAD` 框架需要调用 `BaseMethod` 的接口，继承可以确保 `MTSExample` 兼容框架的设计。

### 示例代码
以下是一个简单的继承示例：

```python
class BaseMethod:
    def __init__(self):
        self.name = "BaseMethod"

    def common_method(self):
        print("This is a method from BaseMethod.")

class MTSExample(BaseMethod):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.name = "MTSExample"

    def specific_method(self):
        print("This is a method specific to MTSExample.")

# 使用示例
example = MTSExample()
example.common_method()  # 调用父类方法
example.specific_method()  # 调用子类方法
```

在你的代码中，`MTSExample` 继承了 `BaseMethod`，因此可以使用 `EasyTSAD` 框架中 `BaseMethod` 提供的接口，同时实现自己的逻辑。