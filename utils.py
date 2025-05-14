import os

def is_valid_image(path):
    valid_ext = ['.jpg', '.jpeg', '.png']
    return os.path.exists(path) and os.path.splitext(path)[1].lower() in valid_ext
