from queue import Queue
from typing import Tuple

from model_engine_server.inference.tool_completion.base import BaseTool

NAME_ERROR_PATTERN = r"NameError: name \\?'([^']+)\\?' is not defined"

PRINT_PATTERN = r"print\(.+?\)"

# Most common imports used during code execution
FIX_ERRORS_MAPPING = {
    "math": "import math",
    "np": "import numpy as np",
    "cmath": "import cmath",
    "norm": "from scipy.stats import norm",
    "plt": "import matplotlib.pyplot as plt",
    "sp": "import sympy as sp",
    "sympy": "import sympy",
    "sqrt": "from cmath import sqrt",
    "erfinv": "from scipy.special import erfinv",
    "t": "from scipy.stats import t",
    "comb": "from scipy.special import comb",
    "Fraction": "from fractions import Fraction",
    "st": "import steam_table as st",
    "pd": "import pandas as pd",
    "stats": "import scipy.stats as stats",
    "opt": "import scipy.optimize as opt",
    "Counter": "from collections import Counter",
    "datetime": "import datetime",
    "gcd": "from fractions import gcd",
    "pi": "from math import pi",
    "quad": "from scipy.integrate import quad",
    "fsolve": "from scipy.optimize import fsolve",
    "factorial": "from math import factorial",
    "tan": "from math import tan",
    "log": "from math import log",
    "symbols": "from sympy import symbols, sin, cos",
    "integrate": "from sympy import symbols, integrate",
    "diff": "from sympy import symbols, sin, cos, diff",
    "sin": "from sympy import symbols, sin, cos",
    "cos": "from sympy import symbols, sin, cos",
    "time": "import time",
    "Symbol": "from sympy import Symbol",
}


# Check if a model response indicates it could be starting a tool
def check_streaming_tool_start(stream_queue: Queue, tool: BaseTool) -> bool:
    # If the queue is empty, we can't start the tool
    if stream_queue.qsize() == 0:
        return False

    # Create the full string from the queue
    queue_str = ""
    for response in list(stream_queue.queue):
        queue_str += response.output.text

    # Check if the start token is in the queue
    if tool.tool_context_start in queue_str:
        return True

    return False


def check_either_substr(str1: str, str2: str) -> bool:
    return str1 in str2 or str2 in str1


# Check if some responses from the queue should be returned
def get_responses_to_yield(
    stream_queue: Queue, tool: BaseTool, tool_started: bool
) -> Tuple[Queue, Queue]:
    """We return a tuple, (responses_to_yield, stream_queue) based on what should be returned"""
    # If we've started the tool, we shouldn't yield anything
    if tool_started:
        return Queue(), stream_queue

    # Otherwise, we should yield everything in the queue that *can't* be part of the start of a tool
    concatenated_queue_str = ""
    responses_to_yield = Queue()  # These are values we're sure we want to return right now
    undecided_queue = (
        Queue()
    )  # These are values that could be part of start token but we aren't sure yet

    # Iterate through the queue and add to the concatenated queue string
    while stream_queue.qsize() > 0:
        response = stream_queue.get()

        # First check if the adding the current response could be part of the start token
        if check_either_substr(
            concatenated_queue_str + response.output.text, tool.tool_context_start
        ):
            # If so, add it to the undecided queue
            undecided_queue.put(response)
            concatenated_queue_str += response.output.text

        # Otherwise, we are confident that everything in the undecided *can't* be part of the start token
        # in addition to the concatenated queue string
        else:
            while not undecided_queue.empty():
                responses_to_yield.put(undecided_queue.get())

            responses_to_yield.put(response)
            concatenated_queue_str = ""

    # Finally, return the responses to yield and the new stream queue
    return responses_to_yield, undecided_queue
