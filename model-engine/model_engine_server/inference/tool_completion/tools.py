import re
import subprocess
from enum import Enum
from typing import Optional, Tuple

import docker
from model_engine_server.inference.tool_completion.base import BaseTool
from model_engine_server.inference.tool_completion.utils import (
    FIX_ERRORS_MAPPING,
    NAME_ERROR_PATTERN,
    PRINT_PATTERN,
)
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", legacy=False)
MAX_CODEBLOCK_RETRIES = 3


class CodeBlockEvaluator(BaseTool):
    """
    A evaluator to "pseudo-safely" execute python code blocks.
    Executes code from a model generated response using a safe python interpreter.
    the code should have the following format:

    ```python
    {code}
    ```
    {output}
    >>>

    The output will be replaced with the output from executing the code.
    """

    tool_context_start = "```python\n"
    tool_call_token = "\n```\n"
    tool_context_end = "\n>>>\n"

    @staticmethod
    def _cleanup_code_error(error_code: str) -> str:
        """This function will clean up an error code from code execution

        Args:
            error_code (str): The full error code (e.g. like below):

            Command '['python', '-c', 'import math\nx = 2\nmath.sqrt(y)']' in image 'continuumio/anaconda3'
              returned non-zero exit status 1: b'Traceback (most recent call last):
              File "<string>", line 3, in <module>\nNameError: name \'y\' is not defined\n'

        Returns:
            str: like the following:

            Traceback (most recent call last): File "<string>", line 3, in <module>
            NameError: name \'y\' is not defined

        """
        if "Traceback" not in error_code:
            return error_code

        # Let's find the byte string: (e.g. b')
        stacktrace = error_code.split("b'")[-1]

        # Now read it as a bytestring
        stacktrace = "\n" + stacktrace.encode("utf-8").decode("unicode_escape")

        return stacktrace.strip("'")

    def __init__(self):
        # Condition to check if we can use docker
        try:
            self.client = docker.from_env()
            self.evaluate = self.evaluate_code_in_docker
        except docker.errors.DockerException:
            # If docker is not available, use the python interpreter
            self.evaluate = self.evaluate_code_using_exec

    def __call__(
        self,
        expression: str,
        past_context: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Given an expression, extract the code block and execute it using a safe python interpreter. Additionally,
        approximate the number of tokens added to the expression from the tool output along with handling retries
        due to simple tool errors (e.g. import errors, missing variables)

        Args:
            expression (str): text with natural language and code blocks
            past_context (Optional[str]): previously generated code blocks for retrying simple code errors

        Returns:
            str: Formatted output from the code execution tool
            int: Number of tokens added

        Raises:
            RuntimeError: If any errors occur during the code execution or retries for simple code errors.
        """
        tool_output = ""
        expression_ = expression
        num_tokens = 0
        if (CodeBlockEvaluator.tool_context_start in expression) and (
            CodeBlockEvaluator.tool_call_token in expression
        ):
            # Extract the expression between the start token and the special token for the tool to evaluate
            code_expression = expression.split(CodeBlockEvaluator.tool_context_start)[-1].split(
                CodeBlockEvaluator.tool_call_token
            )[0]

            # Note: Can increase max retries if needed (e.g. > 1 import errors + variable not defined in code_expression)
            for retry_count in range(MAX_CODEBLOCK_RETRIES):
                try:
                    tool_output = self.evaluate(code_expression)
                    break
                except Exception as e:
                    name_error = re.search(NAME_ERROR_PATTERN, str(e))
                    if (
                        past_context is None
                        or name_error is None
                        or retry_count == MAX_CODEBLOCK_RETRIES - 1
                    ):
                        error_code = self._cleanup_code_error(str(e))
                        raise RuntimeError(f"failed with error: {error_code}")

                    if retry_count == 0 and past_context != "":
                        # Grab all the prior code blocks in "```python\n{code}\n```\n" format
                        code_expression = (
                            self._extract_code_blocks(past_context) + "\n" + code_expression
                        )
                    else:
                        current_error = name_error.group(1).replace("\\", "")
                        # Make sure error is one of the fixable/common import errors seen in the past
                        if current_error not in FIX_ERRORS_MAPPING.keys():
                            error_code = self._cleanup_code_error(str(e))
                            raise RuntimeError(
                                f"failed on retry: {retry_count}, NameError variable: {current_error}, and error: {error_code}"
                            )

                        code_expression = FIX_ERRORS_MAPPING[current_error] + "\n" + code_expression

            tool_output = (
                CodeBlockEvaluator.tool_call_token
                + tool_output
                + CodeBlockEvaluator.tool_context_end
            )

            expression_ = expression.split(CodeBlockEvaluator.tool_call_token)[0] + tool_output
            num_tokens = max(
                0, len(tokenizer(expression_).input_ids) - len(tokenizer(expression).input_ids)
            )
        return expression_, num_tokens

    def _extract_code_blocks(self, context: str):
        """
        Given some text (e.g. previous completion), extract all the code blocks in the format
        along with removing any old print statements.

        Args:
            context (str): text with natural language and code blocks

        Returns:
            str: Parsed code blocks with print statements removed
        """
        code_block_pattern = re.compile(
            rf"{CodeBlockEvaluator.tool_context_start}(.*?){CodeBlockEvaluator.tool_call_token}",
            re.DOTALL,
        )
        code_block_matches = code_block_pattern.findall(context)
        # Remove lines with print statements bc already included in model response
        cleaned_code_blocks = []
        for code_block in code_block_matches:
            no_print_code_blocks = []
            for line in code_block.split("\n"):
                # Ignore lines with print format
                if re.search(PRINT_PATTERN, line) is None:
                    no_print_code_blocks.append(line)
            cleaned_code_blocks.append("\n".join(no_print_code_blocks))
        return "\n".join(cleaned_code_blocks)

    def evaluate_code_in_docker(self, code: str) -> str:
        """
        Executes a block of code using a safe python interpreter and returns the output as a string.

        This function uses a docker container to safely execute a given block of code.
        The function returns the output of the last executed line, if any.

        Args:
            code (str): A string containing the Python code to be executed.

        Returns:
            str: The output of the executed code, converted to string. If there's no explicit output,
                the function returns the result of the last line of code.

        Raises:
            RuntimeError: If any errors occur during the code execution.
        """

        try:
            output = self.client.containers.run(
                "continuumio/anaconda3", command=["python", "-c", code]
            ).decode()
            output = output.strip()
        except docker.errors.ContainerError as e:
            raise RuntimeError(e)

        return output

    def evaluate_code_using_exec(self, code: str) -> str:
        """
        Executes a block of code using the python "exec" function. Returns the output as a string.
        Unfortunately it doesn't have the same safety guarantees as the docker version.

        However, it will only ever be enabled when we are in a scale environment as we check the llmengine
        path.

        Args:
            code (str): A string containing the Python code to be executed.

        Returns:
            str: The output of the executed code, converted to string. If there's no explicit output,
                the function returns the result of the last line of code.
        """
        try:
            p = subprocess.run(["python", "-c", code], capture_output=True, text=True)
            p.check_returncode()  # Raises CalledProcessError if the exit code is non-zero
            output_str = p.stdout

            # If output is empty and the last line didn't have a print statement, edit the code to add one
            if output_str == "" and "print" not in code.split("\n")[-1]:
                new_code = "\n".join(code.split("\n")[:-1])
                last_line = code.split("\n")[-1]
                new_code = new_code + f"\nprint({last_line})"

                # Re-run it
                p = subprocess.run(["python", "-c", new_code], capture_output=True, text=True)
                p.check_returncode()
                output_str = p.stdout

        except subprocess.CalledProcessError as e:
            raise RuntimeError(p.stderr) from e

        return output_str


class Tools(str, Enum):
    CODE_EVALUATOR = "code_evaluator"


TOOL_MAP = {
    Tools.CODE_EVALUATOR: CodeBlockEvaluator,
}
