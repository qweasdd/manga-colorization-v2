import os
import gradio as gr
from datetime import datetime
from PIL import Image

def get_unique_save_path(save_path):
    base, ext = os.path.splitext(save_path)
    counter = 1
    while os.path.exists(save_path):
        save_path = f"{base}_{counter}{ext}"
        counter += 1
    return save_path

def print_cli(image, output=None, gpu=False, no_denoise=False, denoiser_sigma=25, size=576):
    # Crear un directorio temporal para guardar la imagen coloreada si no se proporciona un output
    if output is None or output.strip() == "":
        current_date = datetime.now().strftime('%Y-%m-%d')
        output = os.path.join('.', 'colored', current_date)
        if not os.path.exists(output):
            os.makedirs(output)

    base_name = os.path.basename(image)
    colorized_image_name = os.path.splitext(base_name)[0] + '_colorized.png'
    colorized_image_path = os.path.join(output, colorized_image_name)
    colorized_image_path = get_unique_save_path(colorized_image_path)

    # Construir el comando con los parámetros opcionales
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

    # Verificar que la imagen coloreada existe antes de devolver la ruta
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

def run_interface():
    demo = gr.Blocks()

    with demo:
        with gr.Tab("Colorize Single Image"):
            iface1 = gr.Interface(
                fn=lambda *args: load_image(*args),
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

    demo.launch()

if __name__ == "__main__":
    run_interface()
