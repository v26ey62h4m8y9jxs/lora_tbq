from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch

model_id = "/data2/trained_model/dreamlike-photoreal-2.0-convert"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda"
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

prompt = "a sks woman,protrait,face in the center cyberpunk futuristic neon. by ismail inceoglu dragan bibin hans thoma greg rutkowski alexandros pyromallis nekro rene maritte illustrated,  fine details, realistic shaded"
# torch.manual_seed(0)
# image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]


from lora_diffusion import tune_lora_scale, patch_pipe, safe_open
from lora_diffusion.lora import parse_safeloras

# 保存safetensors到webui可兼容
f_path = '/data2/code/stable-diffusion-webui-newst/extensions/sd-webui-additional-networks/models/lora/lora_weight128.safetensors'
safeloras = safe_open(f_path, framework="pt", device="cpu")
loras = parse_safeloras(safeloras)

## 保存textencoder相关lora参数
text_enc = loras['text_encoder']
i = 0  # layer idx
du_state_dict = {}
print('text_enc target modules:', text_enc[2])
while len(text_enc[0]) > 0:
    for idx in ['k', 'v', 'q', 'out']:
        weight = text_enc[0].pop(0)
        key = 'lora_te_text_model_encoder_layers_%d_self_attn_%s_proj.lora_up.weight' % (i, idx)
        du_state_dict[key] = weight
        weight = text_enc[0].pop(0)
        key = 'lora_te_text_model_encoder_layers_%d_self_attn_%s_proj.lora_down.weight' % (i, idx)
        du_state_dict[key] = weight
        alpha = text_enc[1].pop(0)
        key = 'lora_te_text_model_encoder_layers_%d_self_attn_%s_proj.alpha' % (i, idx)
        du_state_dict[key] = alpha
    i += 1

## 保存unet相关lora参数
unet = loras['unet']
i = 0  # layer idx
print('unet target modules:', unet[2])
while len(unet[0]) > 0:
    for idx in ['q', 'k', 'v', 'out']:
        weight = text_enc[0].pop(0)
        key = 'lora_te_text_model_encoder_layers_%d_self_attn_%s_proj.lora_up.weight' % (i, idx)
        du_state_dict[key] = weight
        weight = text_enc[0].pop(0)
        key = 'lora_te_text_model_encoder_layers_%d_self_attn_%s_proj.lora_down.weight' % (i, idx)
        du_state_dict[key] = weight
        alpha = text_enc[1].pop(0)
        key = 'lora_te_text_model_encoder_layers_%d_self_attn_%s_proj.alpha' % (i, idx)
        du_state_dict[key] = alpha
    i += 1



patch_pipe(
    pipe,
    "/data2/code/stable-diffusion-webui-newst/extensions/sd-webui-additional-networks/models/lora/lora_weight128.safetensors",
    patch_text=True,
    patch_ti=True,
    patch_unet=True,
)

tune_lora_scale(pipe.unet, 0.8)
tune_lora_scale(pipe.text_encoder, 0.8)

torch.manual_seed(10)
image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
image.save("../contents/lion_illust.jpg")
image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
image.save("../contents/tbq1.jpg")
image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
image.save("../contents/tbq2.jpg")