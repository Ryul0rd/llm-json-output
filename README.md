# llm-json-output

This is a rough draft of what I hope to eventually make a feature in huggingface transformers.

## The problem

As people start to create pipelines/agents with LLMs, we want to get structured output from them somehow.
Right now, people do this by creating a prompt that asks an instruct/assistant model to output JSON and hoping for the best.
If the model fails to produce JSON, or produces JSON with the wrong schema, you're out of luck.
Also, you need to spend a decent chunk of your context length specifying the schema.

## The solution

It's possible to actually force the model to only output valid json in the valid schema.
We create a deterministic finite-state automata (DFA) based on our desired schema then, every time we want to generate a token,
for each token in the vocabulary, we check to see if it's valid according to the DFA.
When we sample a token from the model output, we only sample from valid tokens.

## Issues with current implementation

1. Code is just a mess. I had no plan going in so I just solved problems as they came up. Should get a full rewrite.
2. Performance is awful. Many seconds (1-20) are necessary to run on my machine depending on schema vs just 0.2 seconds for freeform generation. Based on my profiling, most of this seems to be an issue with Python just being very slow, rather than poor algorithmic choices. Getting things to run at a decent speed will probably just need an implementation in a faster language like c++ or rust.
3. No ability to add additional constraints beyond type. eg. no limits on string or array length so you may not get a full JSON string.
