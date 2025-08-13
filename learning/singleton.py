class SingletonMeta(type):
    _instances = (
        {}
    )  # dictionary as class attribute that stores: class_name -> class_instance

    def __call__(
        cls, *args, **kwargs
    ):  # triggered when trying to create a new instance of a class
        if (
            cls not in cls._instances
        ):  # we only want to create a new instance if there doesn't already exist one
            cls._instances[cls] = super(SingletonMeta, cls).__call__(
                *args, **kwargs
            )  # call the parent's class (type) __call__ method to create a new object with
            # the args/kwargs and store a mapping from class_name to class_instance in the dictionary
        return cls._instances[
            cls
        ]  # always return the stored instance for a given class


if __name__ == "__main__":

    class Pie(metaclass=SingletonMeta):
        def __init__(self, name):
            self.name = name

    class Cake(metaclass=SingletonMeta):
        def __init__(self, name):
            self.name = name

    chocalate = Pie("choclate")
    strawberry = Pie("strawberry")
    print(chocalate.name)  # should print: chocolate
    print(strawberry.name)  # should print: chocolate

    vanilla = Cake("vanilla")
    fruit = Cake("fruit")
    print(vanilla.name)  # should print: vanilla
    print(fruit.name)  # should print: vanilla
