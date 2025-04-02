import contextvars
from typing import Any, Optional, Type, TypeVar, overload

T = TypeVar("T")

class DependencyContainer:
    """
    作用域容器，提供依赖注册和解析的核心功能。你可以在此获取一些系统的对象

    Attributes:
        parent (DependencyContainer): 父容器实例，用于支持作用域嵌套
        registry (dict): 存储当前容器注册的值或对象实例，格式为{key: value|object}

        docs: https://docs.python.org/zh-cn/3.13/library/contextvars.html#module-contextvars

    Methods:
        register: 向容器注册一个key-value对
        resolve: 从容器解析获取一个值或对象实例
        destroy: 从容器中移除一个值或对象实例
        scoped: 创建一个新的作用域容器
    """
    def __init__(self, parent=None):
        self.parent = parent  # 父容器，用于支持作用域嵌套
        self.registry = {}  # 当前容器的注册表

    def register(self, key, value):
        """
        向容器注册一个值或者实例。


        Args:
            key: 对象的标识键, 为字典的键类型通常为一个字符串
            value: 值/对象实例
        """
        self.registry[key] = value

    @overload
    def resolve(self, key: Type[T]) -> T: ...

    @overload
    def resolve(self, key: Any) -> Any: ...

    def resolve(self, key: Type[T] | Any) -> T | Any:
        """
        依照{key}从容器获取一个值或对象实例。
        如果{key}在当前容器中不存在，则会递归查找父容器。

        Args:
            key: 对象的标识键

        Returns:
            值/对象实例

        Raises:
            KeyError: {key}在当前容器和父容器中都不存在时抛出
        """
        if key in self.registry:
            return self.registry[key]

        elif self.parent:
            return self.parent.resolve(key)
        else:
            raise KeyError(f"Dependency {key} not found.")


    def has(self, key: Type[T] | Any) -> bool:
        """
        检测容器中是否能解析出某个键所对应的值。
        Args:
            key: 对象的标识键
        Returns:
            成功返回 True，失败返回 False
        """
        return key in self.registry or (self.parent is not None and self.parent.has(key))

    @overload
    def destroy(self, key: Type[T]) -> None: ...

    @overload
    def destroy(self, key: Any) -> None: ...

    def destroy(self, key: Type[T] | Any) -> None:
        """
        从容器中移除一个值或对象实例。

        Args:
            key: 对象的标识键
        Raises:
            KeyError: {key}在当前容器和父容器中都不存在时抛出
        """
        if key in self.registry:
            del self.registry[key]
        elif self.parent:
            self.parent.destroy(key)
        else: 
            raise KeyError(f"Cannot destroy dependency {key} which is not found in registry or parent container's registry.")


    def scoped(self):
        """创建一个新的作用域容器"""
        new_container = ScopedContainer(self)

        if DependencyContainer in self.registry:
            new_container.registry[DependencyContainer] = new_container
            new_container.registry[ScopedContainer] = new_container
        return new_container


# 使用 contextvars 实现线程和异步安全的上下文管理
current_container = contextvars.ContextVar[Optional[DependencyContainer]]("current_container", default=None)

class ScopedContainer(DependencyContainer):
    def __init__(self, parent):
        super().__init__(parent)

    def __enter__(self):
        # 将当前容器设置为新的作用域容器
        self.token = current_container.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # 恢复之前的容器
        current_container.reset(self.token)
