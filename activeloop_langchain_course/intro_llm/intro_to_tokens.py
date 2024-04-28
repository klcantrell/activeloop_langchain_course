from transformers import AutoTokenizer


def run():
    # Download and load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # print(tokenizer.vocab)

    token_ids = tokenizer.encode("This is a sample text to test the tokenizer.")

    print("Tokens:   ", tokenizer.convert_ids_to_tokens(token_ids))
    print("Token IDs:", token_ids)
