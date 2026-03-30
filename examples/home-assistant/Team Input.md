LFM2: 

Function definition: LFM2 takes JSON function definitions as input (JSON objects between <|tool_list_start|> and <|tool_list_end|> special tokens), usually in the system prompt

Function call: LFM2 writes Pythonic function calls (a Python list between <|tool_call_start|> and <|tool_call_end|> special tokens), as the assistant answer.

Function execution: The function call is executed and the result is returned (string between <|tool_response_start|> and <|tool_response_end|> special tokens), as a "tool" role.

Final answer: LFM2 interprets the outcome of the function call to address the original user prompt in plain text.


<|startoftext|><|im_start|>system
List of tools: <|tool_list_start|>[{"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}]<|tool_list_end|><|im_end|>
<|im_start|>user
What is the current status of candidate ID 12345?<|im_end|>
<|im_start|>assistant
<|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking the current status of candidate ID 12345.<|im_end|>
<|im_start|>tool
<|tool_response_start|>[{"candidate_id": "12345", "status": "Interview Scheduled", "position": "Clinical Research Associate", "date": "2023-11-20"}]<|tool_response_end|><|im_end|>
<|im_start|>assistant
The candidate with ID 12345 is currently in the "Interview Scheduled" stage for the position of Clinical Research Associate, with an interview date set for 2023-11-20.<|im_end|>

LFM2.5

LFM2.5 supports function calling as follows:

Function definition: We recommend providing the list of tools as a JSON object in the system prompt. You can also use the tokenizer.apply_chat_template() function with tools.

Function call: By default, LFM2.5 writes Pythonic function calls (a Python list between <|tool_call_start|> and <|tool_call_end|> special tokens), as the assistant answer. You can 
override this behavior by asking the model to output JSON function calls in the system prompt.

Function execution: The function call is executed, and the result is returned as a "tool" role.

Final answer: LFM2 interprets the outcome of the function call to address the original user prompt in plain text.


<|startoftext|><|im_start|>system
List of tools: [{"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}]<|im_end|>
<|im_start|>user
What is the current status of candidate ID 12345?<|im_end|>
<|im_start|>assistant
<|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking the current status of candidate ID 12345.<|im_end|>
<|im_start|>tool
[{"candidate_id": "12345", "status": "Interview Scheduled", "position": "Clinical Research Associate", "date": "2023-11-20"}]<|im_end|>
<|im_start|>assistant
The candidate with ID 12345 is currently in the "Interview Scheduled" stage for the position of Clinical Research Associate, with an interview date set for 2023-11-20.<|im_end|>

General issues; 


Parsing 

It originally scored 0.640 because it outputs bracket notation:
[get_weather(city="Antwerp")]
instead of XML tool tags. After fixing the parser, it turned out the model had been making correct decisions all along.
It requires a bracket-notation parser (not standard <tool_call> tags).
Inference
for sglang you can do
sglang serve --model-path LiquidAI/LFM2.5-1.2B-Instruct --tool-call-parser lfm2
For vllm: 
vllm serve <model> \
    --enable-auto-tool-choice \
    --tool-call-parser <parser_name> \
    --chat-template <path_to_template>  # optional for some parsers

vLLM doesn't have a built-in parser for LFM2's format, but since LFM2 uses Pythonic tool calls (similar to the pythonic parser format), you have two options:
Try the pythonic parser - It may work with minor template adjustments since LFM2 uses Python list syntax for calls
Write a custom tool parser plugin - Use --tool-parser-plugin to register a parser that handles LFM2's specific <<|tool_call_start|>> / <<|tool_call_end|>> tokens
Given LFM2's unique special tokens, you'd likely need a custom parser or chat template to fully support it in vLLM's OpenAI-compatible server.
Tool calling shall work in llama.cpp without additional parameters, given gguf has the correct chat template that supports tool calling
"Output function calls as JSON"
Function call: By default, LFM2.5 writes Pythonic function calls (a Python list between <|tool_call_start|> and <|tool_call_end|> special tokens), as the assistant answer. You can override this behavior by asking the model to output JSON function calls in the system prompt.
