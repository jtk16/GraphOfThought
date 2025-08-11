from transformers import AutoModelForCausalLM, AutoTokenizer

def load_mistral_model(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Loads the Mistral-7B-Instruct model and its tokenizer.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        print(f"Successfully loaded {model_name}")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None

if __name__ == "__main__":
    tokenizer, model = load_mistral_model()
    if model and tokenizer:
        print("Model and tokenizer loaded successfully.")
        # Example usage:
        # prompt = "Hello, my name is"
        # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # outputs = model.generate(**inputs, max_new_tokens=20)
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
