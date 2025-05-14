from image_processor import detect_objects
from text_generator import generate_text
from utils import is_valid_image

def main():
    image_path = input("Enter path to image: ").strip()
    prompt = input("Enter your text prompt: ").strip()

    if not is_valid_image(image_path):
        print("Invalid image file. Please use .jpg/.jpeg/.png.")
        return

    if not prompt:
        print("Prompt cannot be empty.")
        return

    print("\nDetecting objects...")
    objects = detect_objects(image_path)

    if not objects:
        print("No objects detected.")
    else:
        print("Detected objects:")
        for obj in objects:
            print(f"- {obj['name']} ({obj['confidence']})")

    print("\nGenerating response...")
    response = generate_text(prompt, objects)
    print("\n=== Final Response ===")
    print(response)

if __name__ == "__main__":
    main()
