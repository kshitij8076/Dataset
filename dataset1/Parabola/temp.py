import os

def rename_png_images(directory='.'):
    png_files = [f for f in os.listdir(directory) if f.lower().endswith('.png')]
    png_files.sort()  # Sort to ensure consistent numbering

    for index, filename in enumerate(png_files, start=1):
        new_name = f"Parabola_{index}.png"
        src = os.path.join(directory, filename)
        dst = os.path.join(directory, new_name)
        os.rename(src, dst)
        print(f"Renamed: {filename} -> {new_name}")

if __name__ == "__main__":
    rename_png_images()
