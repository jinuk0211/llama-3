#llama 3 from scrath 한국어 번역
=============
토크나이져
---------------------
![image](https://github.com/jinuk0211/llama_3_-/assets/150532431/2c1343bc-40d1-49bb-acad-4c6fab6ed385)

```python
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import matplotlib.pyplot as plt

tokenizer_path = "Meta-Llama-3-8B/tokenizer.model"
special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)
```
's (is or has)
't (not)
're (are)
've (have)
'm (am)
'll (will)
'd (would or had)

[^\r\n\p{L}\p{N}]?\p{L}
캐리지 리턴(\r), 새 줄(\n), 문자(\p{L}), 숫자(\p{N})가 아닌 문자가 0개 또는 1개 온 뒤에 하나 이상의 문자(\p{L})가 오는 경우

\p{N}{1,3}
1~3자리의 숫자(\p{N})를 매칭

?[^\s\p{L}\p{N}]+[\r\n]*
공백이 0개 또는 1개 온 뒤, 공백(\s), 문자(\p{L}), 숫자(\p{N})가 아닌 문자가 1개 이상 오고, 그 뒤에 캐리지 리턴 또는 새 줄이 0개 이상 오는 경우를 매칭

\s*[\r\n]+
0개 이상의 공백(\s*)이 온 뒤에 하나 이상의 캐리지 리턴 또는 새 줄([\r\n]+)이 오는 경우를 매칭

\s+(?!\S)
이 부분은 하나 이상의 공백(\s+)이 오고, 그 뒤에 비공백 문자(\S)가 오지 않는 경우를 매칭

\s+
이 부분은 하나 이상의 공백(\s+)을 매칭

```python
model = torch.load("Meta-Llama-3-8B/consolidated.00.pth")
print(json.dumps(list(model.keys())[:20], indent=4))
```
```
