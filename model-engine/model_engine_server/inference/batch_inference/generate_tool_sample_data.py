import json

COMPLETION_PROMPT1 = """\
FYI: you can write code like this: 
```python
import math
print(math.sqrt(2))
```
1.41...
>>>

For reference, the third digit of 4.32 is 2. Also, use "Final Answer: X" to indicate your final answer.

### Problem:

What is the 4th digit of pi?

### Answer:
```python
import math
print(math.pi)
```
3.141592653589793
>>>

Final Answer: 1

### Problem:

What is the 4th digit of the square root of 2?

### Answer: 
"""

COMPLETION_PROMPT2 = """\
FYI: you can write code like this: 
```python
import math
print(math.sqrt(2))
```
1.41...
>>>

For reference, the third digit of 4.32 is 2. Also, use "Final Answer: X" to indicate your final answer.

### Problem:

What is the 4th digit of pi?

### Answer:
```python
import math
print(math.pi)
```
3.141592653589793
>>>

Final Answer: 1

### Problem:

What is the 5th digit of the square root of 2?

### Answer: 
"""

data = {
    "prompts": [
        COMPLETION_PROMPT1,
        COMPLETION_PROMPT2,
        "what is deep learning",
    ],
    "max_new_tokens": 100,
    "temperature": 0.0,
    "return_token_log_probs": True,
    "stop_sequences": ["</s>", "\n### Problem:\n", ">>>\n"],
}

json.dump(data, open("sample_data_tool.json", "w"))
