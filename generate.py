import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionImg2ImgPipeline, AutoencoderKL, DDPMScheduler

# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# from src.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_replace_text_with_img import StableDiffusionReplaceTextWithImagePipeline
from transformers import ViTModel, CLIPTokenizer, CLIPFeatureExtractor
from diffusers.utils import make_image_grid, load_image
from PIL import Image
import torch
import random
from tqdm import tqdm

from config import cfg
from model import make_model

def random_noise(image):
    h, w = image.height, image.width
    pixels = []
    for y in range(h):
        row_pixels = []
        for x in range(w):
            # random rgb（0-255）
            r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            row_pixels.append((r, g ,b))
        pixels.append(row_pixels)
    image.putdata([pixel for row in pixels for pixel in row])
    return image

def retrieve_timesteps(
    scheduler,
    num_inference_steps = None,
    device = None,
    timesteps = None,
    **kwargs,
):
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

if __name__ == '__main__':
    dir = "outputs"
    if not os.path.exists(dir):
        os.makedirs(dir)
        

    # unet = UNet2DConditionModel.from_pretrained("/home/liyuke/data/codes/diffusers/examples/text_to_image/sd-sysuIR-256-128-model/checkpoint-20000/unet", torch_dtype=torch.float16)
    SD_path = "runwayml/stable-diffusion-v1-5"
    unet = UNet2DConditionModel.from_pretrained("/home/liyuke/data/exp/diffusion_vit_base16/Market/unet", torch_dtype=torch.float16).to('cuda')
    vae = AutoencoderKL.from_pretrained(SD_path, subfolder="vae", torch_dtype=torch.float16).to('cuda')
    # tokenizer = CLIPTokenizer.from_pretrained(SD_path, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(SD_path, subfolder="scheduler")
    # feature_extractor = CLIPFeatureExtractor(SD_path, subfolder="feature_extractor")

    ### img2img with text guidance
    url = "/home/liyuke/data/market1501/bounding_box_train/0184_c3s2_147869_05.jpg"
    init_image = load_image(url)
    print(type(init_image))
    import torchvision.transforms as T
    init_image = T.Resize([256,128])(init_image)


    init_image.save("{}/A_origin_image.png".format(dir))
    # init_image = [init_image]*64


    ### text2img
    cfg.merge_from_file("config/diffusion_vit.yml")
    vit = make_model(cfg=cfg, modelname="vit", num_class=0).to('cuda').to(torch.float16)
    vit.load_param("/home/liyuke/data/exp/diffusion_vit_base16/Market/vit_best.pth")
    vit.eval()


    ##### to do: img2img_pipe #####
    noise = torch.randn([16,4,256//8,128//8], device='cuda', dtype=torch.float16)
    with torch.no_grad():
        image_tensors = T.ToTensor()(init_image).unsqueeze(0).repeat(16,1,1,1).to('cuda').to(noise.dtype)
        img_feat = vit(image_tensors).to(noise.dtype).unsqueeze(1)
        # print(img_feat.dtype, noise.dtype)
        timesteps, _ = retrieve_timesteps(scheduler, 50)
        print(timesteps)
        for t in tqdm(scheduler.timesteps):
            noisy_residual = unet(noise, t, img_feat)['sample']
            previous_noisy_sample = scheduler.step(noisy_residual, t, noise).prev_sample
            noise = previous_noisy_sample
        outputs = vae.decode(noise)[0]
    images = (outputs / 2 + 0.5).clamp(0, 1).squeeze()

    for i, image in enumerate(images):
        image = T.ToPILImage()(image)
        image.save("{}/test{}.png".format(dir, i))
        # print(type(image))