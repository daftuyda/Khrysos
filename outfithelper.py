import os
import shutil

def organize_files(source_folder, target_root):
    # Define the mapping of frame numbers to expressions
    expression_ranges = {
        range(0, 4): 'normal',
        range(4, 8): 'surprised',
        range(8, 12): 'confused',
        range(12, 16): 'love',
        range(16, 20): 'happy',
        range(20, 24): 'angry'
    }

    for filename in os.listdir(source_folder):
        if filename.endswith(".png"):
            # Extract outfit name
            outfit = ''.join(filter(str.isalpha, filename.split('.')[0]))

            # Create an outfit folder if it doesn't exist
            outfit_folder = os.path.join(target_root, outfit)
            os.makedirs(outfit_folder, exist_ok=True)

            # Determine the global frame number
            global_frame_number = int(''.join(filter(str.isdigit, filename)))

            # Determine the expression and local frame number
            for frame_range, expression in expression_ranges.items():
                if global_frame_number in frame_range:
                    local_frame_number = global_frame_number - min(frame_range)

                    # New file name format: '{expression}Blink{local_frame_number}.png'
                    new_filename = f'{expression}Blink{local_frame_number}.png'
                    target_file = os.path.join(outfit_folder, new_filename)

                    # Move and rename the file
                    source_file = os.path.join(source_folder, filename)
                    shutil.move(source_file, target_file)
                    print(f"Moved and renamed '{source_file}' to '{target_file}'")

                    # Duplicate the first frame of each expression
                    if local_frame_number == 0:
                        duplicate_filename = f'{expression}.png'
                        duplicate_file = os.path.join(outfit_folder, duplicate_filename)
                        shutil.copy(target_file, duplicate_file)
                        print(f"Copied '{target_file}' to '{duplicate_file}'")

                    break

# Example usage
source_folder = 'D:\Downloads\yuki\Blink'
target_root = 'D:\Void\Khrysos\sprites'
organize_files(source_folder, target_root)