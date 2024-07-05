import os
import requests
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
    esrgan_url = "https://huggingface.co/KaiserQ/Models-GEN/resolve/main/Manga-Colorization-GUI/RealESRGAN_x4plus_anime_6B.pth"

    generator_dest = "networks"
    extractor_dest = "denoising/models"
    esrgan_dest = "weights"

    generator_path = download_file(generator_url, generator_dest)
    extractor_path = download_file(extractor_url, extractor_dest)
    esrgan_path = download_file(esrgan_url, esrgan_dest)

    return f"Downloaded {generator_path}, {extractor_path}, and {esrgan_path}"