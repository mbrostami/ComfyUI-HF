from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Node:
    """
    A node for generating text using GPT-2, with the ability to specify the model via Hugging Face model identifier.

    Attributes
    ----------
    See base class.
    """
    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines input types for the GPT-2 node, including a model identifier.
        """
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True}),
                "hg_gpt2_identifier": ("STRING", {
                    "default": "Gustavosta/MagicPrompt-Stable-Diffusion",
                    "multiline": False
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING",)
    RETURN_NAMES = ("CONDITIONING","STRING",)
    FUNCTION = "generate"
    CATEGORY = "conditioning"

    def generate(self, clip, text, hg_gpt2_identifier, temperature):
        """
        Generates text based on the provided prompt using a specified GPT-2 model.

        Parameters
        ----------
        text : str
            The text prompt to generate text from.
        hg_gpt2_identifier : str
            The Hugging Face model identifier for the GPT-2 model to use.
        temperature : float
            The temperature to use for generating text, controlling randomness.

        Returns
        -------
        tuple
            A tuple containing the generated text.
        """
        num_return_sequences = 1
        max_length = 100 # max: 512
        # Load the model and tokenizer based on the hg_gpt2_identifier
        tokenizer = GPT2Tokenizer.from_pretrained(hg_gpt2_identifier)
        model = GPT2LMHeadModel.from_pretrained(hg_gpt2_identifier)
        model.eval()  # Recommended for inference

        input_ids = tokenizer.encode(text, return_tensors='pt')
        max_length = min(max_length + len(input_ids[0]), 1024)  # Adjusting max_length to account for prompt length

        # Generate text
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            no_repeat_ngram_size=2
        )

        new_prompts = tuple(tokenizer.decode(output, skip_special_tokens=True) for output in outputs)

        # Process each prompt individually
        tokens = clip.tokenize(new_prompts[0])  # Tokenize the individual prompt
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)  # Encode the tokens
        return ([[cond, {"pooled_output": pooled}]], new_prompts[0],)


# Update the NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS to include the GPT2Node
NODE_CLASS_MAPPINGS = {
    "GPT2Node": GPT2Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT2Node": "GPT-2 Text Generator"
}