
from jacinle.utils.printing import kvprint, kvformat


def print_args(args):
    kvprint(args.__dict__)


def format_args(args):
    return kvformat(args.__dict__)

