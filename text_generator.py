from transformers import pipeline

# Load model once
text_gen_pipeline = pipeline("text-generation", model="gpt2")

def generate_text(prompt, object_info):
    try:
        object_list = ", ".join([f"{obj['name']} ({obj['confidence']})" for obj in object_info])
        full_prompt = f"{prompt}\nBased on the image, I detected: {object_list}.\n"

        output = text_gen_pipeline(full_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        return output
    except Exception as e:
        print(f"Text generation error: {e}")
        return "Failed to generate text."
