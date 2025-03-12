from PIL.ImageOps import grayscale

from Preprocess_data import *

input_folder = "D:\FPTUniversity\Capstone_Project\Code_tesst\out"
output_folder = "D:\FPTUniversity\Capstone_Project\Preprocess_Data"

processor = ImagePreprocessor(input_folder, output_folder, grayscale= False)
processor.process_all_images()