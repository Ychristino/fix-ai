from typing import Union, Tuple, Type


class NoSuchFileException(Exception):
    def __init__(self, path: str = None, filename: str = None):
        Exception.__init__(self)
        self.path = path
        self.filename = filename
        self.message = None

    def __str__(self):
        self.message = (f"The file"
                        f" {self.filename if self.filename is not None else ''}"
                        f" was not found in the Path"
                        f" {self.path if self.path is not None else ''}"
                        )

        return self.message


class InvalidInstanceException(Exception):
    def __init__(self, object_type: Type, expected_type: Union[Type, Tuple[Type]]):
        super().__init__(self)
        self.object_type = object_type.__name__
        self.expected_type = expected_type
        self.message = None

    def __str__(self):
        if isinstance(self.expected_type, tuple):
            expected_types_str = ' or '.join(t.__name__ for t in self.expected_type)
        else:
            expected_types_str = self.expected_type.__name__

        self.message = (f"Object of type {self.object_type} not valid! "
                        f"Expected {expected_types_str}.")

        return self.message
