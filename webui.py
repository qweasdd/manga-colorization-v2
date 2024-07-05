import os
import zipfile
import rarfile
import gradio as gr
from datetime import datetime
from PIL import Image
from webui_utils.download import download_weights
from pathlib import Path
import argparse
import cv2
import torch
from realesrgan import RealESRGANer

def get_unique_save_path(save_path):
    base, ext = os.path.splitext(save_path)
    counter = 1
    while os.path.exists(save_path):
        save_path = f"{base}_{counter}{ext}"
        counter += 1
    return save_path

def extract_images_from_archive(archive_path, temp_dir):
    ext = os.path.splitext(archive_path)[1].lower()
    extracted_files = []

    if ext == ".zip":
        with zipfile.ZipFile(archive_path, 'r') as archive:
            archive.extractall(temp_dir)
            extracted_files = [os.path.join(temp_dir, f) for f in archive.namelist() if not f.endswith('/')]
    elif ext in [".cbr", ".cbz"]:
        with rarfile.RarFile(archive_path, 'r') as archive:
            archive.extractall(temp_dir)
            extracted_files = [os.path.join(temp_dir, f) for f in archive.namelist() if not f.endswith('/')]

    return extracted_files

def print_cli(image, output=None, gpu=False, no_denoise=False, denoiser_sigma=25, size=576, upscale=False):
    if output is None or output.strip() == "":
        current_date = datetime.now().strftime('%Y-%m-%d')
        output = os.path.join('.', 'colored', current_date)
        if not os.path.exists(output):
            os.makedirs(output)

    base_name = os.path.basename(image)
    colorized_image_name = os.path.splitext(base_name)[0] + '_colorized.png'
    colorized_image_path = os.path.join(output, colorized_image_name)
    colorized_image_path = get_unique_save_path(colorized_image_path)

    command = f"python inference.py -p \"{image}\" -o \"{output}\""
    if gpu:
        command += " -g"
    if no_denoise:
        command += " -nd"
    if denoiser_sigma:
        command += f" -ds {denoiser_sigma}"
    if size:
        command += f" -s {size}"

    os.system(command)

    if os.path.exists(colorized_image_path):
        if upscale:
            upscale_image(colorized_image_path, colorized_image_path, gpu)
        return colorized_image_path
    else:
        return "Error: No colorized image found."

def upscale_image(input_image_path, output_image_path, gpu):
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    model = RealESRGANer(
        scale=4,
        model_path='weights/RealESRGAN_x4plus_anime_6B.pth',  # Use the anime model
        dni_weight=None,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=device
    )

    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sr_image, _ = model.enhance(image)

    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, sr_image)

def load_image(image_path, output, gpu, no_denoise, denoiser_sigma, size, upscale):
    colorized_image_path = print_cli(image_path, output, gpu, no_denoise, denoiser_sigma, size, upscale)
    if os.path.exists(colorized_image_path):
        return Image.open(colorized_image_path)
    else:
        return None

def colorize_multiple_images(image_paths, output, gpu, no_denoise, denoiser_sigma, size, upscale):
    colorized_images = []
    for image_path in image_paths:
        colorized_image_path = print_cli(image_path, output, gpu, no_denoise, denoiser_sigma, size, upscale)
        if os.path.exists(colorized_image_path):
            colorized_images.append(Image.open(colorized_image_path))
    return colorized_images

def colorize_folder(input_folder, output_folder, gpu, no_denoise, denoiser_sigma, size, upscale):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                image_files.append(file_path)
            elif ext in ['.zip', '.cbr', '.cbz']:
                temp_dir = Path('./temp_extracted')
                if not temp_dir.exists():
                    temp_dir.mkdir(parents=True, exist_ok=True)
                extracted_files = extract_images_from_archive(file_path, temp_dir)
                image_files.extend(extracted_files)

    colorized_images = []
    for image_path in image_files:
        colorized_image_path = print_cli(image_path, output_folder, gpu, no_denoise, denoiser_sigma, size, upscale)
        if os.path.exists(colorized_image_path):
            colorized_images.append(Image.open(colorized_image_path))

    return colorized_images

def colorize_archive(archive_path, output, gpu, no_denoise, denoiser_sigma, size, upscale):
    temp_dir = Path('./temp_extracted')
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = extract_images_from_archive(archive_path, temp_dir)

    colorized_images = []
    for image_path in extracted_files:
        colorized_image_path = print_cli(image_path, output, gpu, no_denoise, denoiser_sigma, size, upscale)
        if os.path.exists(colorized_image_path):
            colorized_images.append(Image.open(colorized_image_path))

    # Optionally re-package colorized images into an archive
    colorized_archive_path = get_unique_save_path(os.path.splitext(archive_path)[0] + "_colorized.zip")
    with zipfile.ZipFile(colorized_archive_path, 'w') as colorized_archive:
        for image_path in colorized_images:
            colorized_archive.write(image_path.filename, os.path.basename(image_path.filename))

    return colorized_archive_path

def run_interface(share=False):
    with gr.Blocks() as demo:
        with gr.Tab("Colorize Single Image"):
            with gr.Row():
                with gr.Column():
                    single_image_interface = gr.Interface(
                        fn=load_image,
                        inputs=[
                            gr.Image(type='filepath', label="Image", elem_classes="input-image", height=500, width=700),
                            gr.Textbox(label="Output Location", placeholder="Optional"),
                            gr.Checkbox(label="Use GPU"),
                            gr.Checkbox(label="No Denoise"),
                            gr.Slider(0, 100, label="Denoiser Sigma", value=25, step=1),
                            gr.Slider(0, 4000, label="Size", value=576, step=32),
                            gr.Checkbox(label="Upscale Image")  # Added checkbox for upscaling
                        ],
                        outputs=gr.Image(type='pil', label="Colorized Image", height=800, width=700)
                    )

        with gr.Tab("Colorize Multiple Images"):
            with gr.Row():
                with gr.Column():
                    multiple_images_interface = gr.Interface(
                        fn=colorize_multiple_images,
                        inputs=[
                            gr.Files(label="Images", type='filepath'),
                            gr.Textbox(label="Output Location", placeholder="Optional"),
                            gr.Checkbox(label="Use GPU"),
                            gr.Checkbox(label="No Denoise"),
                            gr.Slider(0, 100, label="Denoiser Sigma", value=25, step=1),
                            gr.Slider(0, 4000, label="Size", value=576, step=32),
                            gr.Checkbox(label="Upscale Image")  # Added checkbox for upscaling
                        ],
                        outputs=gr.Gallery(label="Colorized Images", columns=4, height="auto")
                    )

        with gr.Tab("Colorize Folder"):
            with gr.Row():
                with gr.Column():
                    folder_interface = gr.Interface(
                        fn=colorize_folder,
                        inputs=[
                            gr.Textbox(label="Input Folder", placeholder="Input folder path"),
                            gr.Textbox(label="Output Folder", placeholder="Output folder path"),
                            gr.Checkbox(label="Use GPU"),
                            gr.Checkbox(label="No Denoise"),
                            gr.Slider(0, 100, label="Denoiser Sigma", value=25, step=1),
                            gr.Slider(0, 4000, label="Size", value=576, step=32),
                            gr.Checkbox(label="Upscale Image")  # Added checkbox for upscaling
                        ],
                        outputs=gr.Gallery(label="Colorized Images", columns=4, height="auto")
                    )

        with gr.Tab("Colorize Archive"):
            with gr.Row():
                with gr.Column():
                    archive_interface = gr.Interface(
                        fn=colorize_archive,
                        inputs=[
                            gr.File(label="Archive (ZIP, CBR, CBZ)", type='filepath'),
                            gr.Textbox(label="Output Location", placeholder="Optional"),
                            gr.Checkbox(label="Use GPU"),
                            gr.Checkbox(label="No Denoise"),
                            gr.Slider(0, 100, label="Denoiser Sigma", value=25, step=1),
                            gr.Slider(0, 4000, label="Size", value=576, step=32),
                            gr.Checkbox(label="Upscale Image")  # Added checkbox for upscaling
                        ],
                        outputs=gr.Textbox(label="Colorized Archive Path")
                    )

        with gr.Tab("Extras"):
            with gr.Row():
                with gr.Column():
                    extras_interface = gr.Interface(
                        fn=download_weights,
                        inputs=[],
                        outputs=gr.Textbox(label="Download Status")
                    )

    demo.launch(share=share)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gradio Interface")
    parser.add_argument('-url', action='store_true', help='Generate a public URL for the Gradio interface')
    args = parser.parse_args()

    run_interface(share=args.url)
