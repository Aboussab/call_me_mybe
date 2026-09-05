from sys import exit, argv
from .parsing import Parsing


if __name__ == "__main__":
    try:
        arg = Parsing()
        arg.parser(argv)
        arg.validation_v1()
    except Exception as e:
        print(e)
        exit(1)
