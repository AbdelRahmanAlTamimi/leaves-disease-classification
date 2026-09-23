import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

data_dir = "/home/tamimi6030/workspace/leaves-disease-classification/plantvillage-dataset/color"


def check_image(args):
    class_name, filepath, filename = args

    try:
        with Image.open(filepath) as img:
            if img.mode != "RGB":
                return (class_name, filename, f"mode={img.mode}")

    except Exception as e:
        return (class_name, filename, f"ERROR: {e}")

    return None


tasks = []

for class_entry in os.scandir(data_dir):
    if not class_entry.is_dir():
        continue

    class_name = class_entry.name

    for file_entry in os.scandir(class_entry.path):
        if file_entry.is_file():
            tasks.append(
                (class_name, file_entry.path, file_entry.name)
            )


issues = []

with ThreadPoolExecutor(max_workers=16) as executor:
    for result in executor.map(check_image, tasks):
        if result is not None:
            issues.append(result)


print(f"Number of problems detected: {len(issues)}")

for issue in issues[:20]:
    print(issue)