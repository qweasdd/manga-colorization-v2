import os
import requests
import gradio as gr
from datetime import datetime
from PIL import Image


def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    response = requests.get(url)
    file_name = url.split("/")[-1]
    file_path = os.path.join(dest_folder, file_name)
    with open(file_path, "wb") as file:
        file.write(response.content)
    return file_path


def download_weights():
    generator_url = "https://huggingface.co/KaiserQ/Models-GEN/resolve/main/Manga-Colorization-GUI/generator.zip"
    extractor_url = "https://huggingface.co/KaiserQ/Models-GEN/resolve/main/Manga-Colorization-GUI/net_rgb.pth"

    generator_dest = "networks"
    extractor_dest = "denoising/models"

    generator_path = download_file(generator_url, generator_dest)
    extractor_path = download_file(extractor_url, extractor_dest)

    return f"Downloaded {generator_path} and {extractor_path}"


def get_unique_save_path(save_path):
    base, ext = os.path.splitext(save_path)
    counter = 1
    while os.path.exists(save_path):
        save_path = f"{base}_{counter}{ext}"
        counter += 1
    return save_path


def print_cli(image, output=None, gpu=False, no_denoise=False, denoiser_sigma=25, size=576):
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
        return colorized_image_path
    else:
        return "Error: No colorized image found."


def load_image(image_path, output, gpu, no_denoise, denoiser_sigma, size):
    colorized_image_path = print_cli(image_path, output, gpu, no_denoise, denoiser_sigma, size)
    if os.path.exists(colorized_image_path):
        return Image.open(colorized_image_path)
    else:
        return None


def colorize_multiple_images(image_paths, output, gpu, no_denoise, denoiser_sigma, size):
    colorized_images = []
    for image_path in image_paths:
        colorized_image_path = print_cli(image_path, output, gpu, no_denoise, denoiser_sigma, size)
        if os.path.exists(colorized_image_path):
            colorized_images.append(Image.open(colorized_image_path))
    return colorized_images


def colorize_folder(input_folder, output_folder, gpu, no_denoise, denoiser_sigma, size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                   os.path.isfile(os.path.join(input_folder, f))]
    colorized_images = []
    for image_path in image_files:
        colorized_image_path = print_cli(image_path, output_folder, gpu, no_denoise, denoiser_sigma, size)
        if os.path.exists(colorized_image_path):
            colorized_images.append(Image.open(colorized_image_path))
    return colorized_images


def run_interface():
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
                            gr.Slider(0, 4000, label="Size", value=576, step=32)
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
                            gr.Slider(0, 4000, label="Size", value=576, step=32)
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
                            gr.Slider(0, 4000, label="Size", value=576, step=32)
                        ],
                        outputs=gr.Gallery(label="Colorized Images", columns=4, height="auto")
                    )

        with gr.Tab("Extras"):
            with gr.Row():
                with gr.Column():
                    extras_interface = gr.Interface(
                        fn=download_weights,
                        inputs=[],
                        outputs=gr.Textbox(label="Download Status")
                    )

    demo.launch()


if __name__ == "__main__":
    run_interface()
