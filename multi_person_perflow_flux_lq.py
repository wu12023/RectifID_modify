import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
import torch.nn.functional as F

import time
import copy
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


import os
# os.environ['http_proxy'] = "" 
# os.environ['https_proxy'] = ""
device_id = 0
torch.cuda.set_device('cuda:%d' % device_id)


from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers import FluxPipeline
from piecewise_rectified_flow.src.scheduler_perflow import PeRFlowScheduler

pipe = FluxPipeline.from_pretrained("./models/flux", safety_checker=None, torch_dtype=torch.float16, local_file_only = True)
# pipe = FluxPipeline.from_pretrained("./models/flux",torch_dtype=torch.float16, local_file_only = True)
# pipe = StableDiffusionPipeline.from_pretrained("hansyan/perflow-sd15-realisticVisionV51", safety_checker=None, torch_dtype=torch.float16)
# pipe = StableDiffusionPipeline.from_pretrained("hansyan/perflow-sd15-disney", safety_checker=None, torch_dtype=torch.float16)
print("load pipe down")
pipe.scheduler = PeRFlowScheduler.from_config(pipe.scheduler.config, prediction_type="diff_eps", num_time_windows=4)
pipe.enable_model_cpu_offload()

for module in [pipe.vae, pipe.text_encoder, pipe.transformer]:
#for module in [pipe.vae, pipe.text_encoder, pipe.unet]:
    for param in module.parameters():
        param.requires_grad = False

pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

my_forward = pipe.__call__.__wrapped__
print(my_forward)

import insightface
from onnx2torch import convert

# antelopev2
# https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo
detector = insightface.model_zoo.get_model('./models/antelopev2/scrfd_10g_bnkps.onnx', provider_options=[{'device_id': device_id}, {}])
detector.prepare(ctx_id=0, input_size=(640, 640))
model = convert('./models/antelopev2/glintr100.onnx').eval().to('cuda')
for param in model.parameters():
    param.requires_grad_(False)




from PIL import Image
import torchvision.transforms.functional as TF
import kornia


ref1 = './character_image/character_2.png'
# ref1 = 'assets/bengio.jpg'
# ref1 = 'assets/schmidhuber.jpg'
# ref1 = 'assets/johansson.jpg'
# ref1 = 'assets/newton.jpg'
# ref2 = 'assets/hinton.jpg'
ref2 = './character_image/character_4.png'
# ref2 = 'assets/schmidhuber.jpg'
# ref2 = 'assets/johansson.jpg'
# ref2 = 'assets/newton.jpg'

ref_image1 = Image.open(ref1).convert("RGB")
ref_image2 = Image.open(ref2).convert("RGB")

def crop_image_embed(ref_image):
    with torch.no_grad():
        det_thresh_backup = detector.det_thresh
        boxes = []
        while len(boxes) == 0:
            boxes, kpss = detector.detect(np.array(ref_image), max_num=1)
            detector.det_thresh -= 0.1
        detector.det_thresh = det_thresh_backup
        M = insightface.utils.face_align.estimate_norm(kpss[0])
        ref_image_cropped = kornia.geometry.transform.warp_affine(
            TF.to_tensor(ref_image).unsqueeze(0).to('cuda'), torch.tensor(M).float().unsqueeze(0).to('cuda'), (112, 112)
        ) * 2 - 1

        ref_embedding = model(ref_image_cropped)
    return ref_image_cropped, ref_embedding

ref_image_cropped1, ref_embedding1 = crop_image_embed(ref_image1)
ref_image_cropped2, ref_embedding2 = crop_image_embed(ref_image2)

cropped_image1 = np.array((ref_image_cropped1[0] / 2 + 0.5).cpu().permute(1, 2, 0) * 255, dtype=np.uint8)
cropped_image2 = np.array((ref_image_cropped2[0] / 2 + 0.5).cpu().permute(1, 2, 0) * 255, dtype=np.uint8)



import tensorflow as tf
from deepface import DeepFace

tf.config.set_visible_devices([], device_type='GPU')
attribute1 = DeepFace.analyze(img_path=ref1, actions = ['gender', 'race'])
attribute2 = DeepFace.analyze(img_path=ref2, actions = ['gender', 'race'])
print(attribute1)
print(attribute2)


idx1 = np.argmax([a['region']['w'] * a['region']['h'] for a in attribute1])
print(attribute1[idx1]['dominant_gender'], attribute1[idx1]['dominant_race'])

idx2 = np.argmax([a['region']['w'] * a['region']['h'] for a in attribute2])
print(attribute2[idx2]['dominant_gender'], attribute2[idx2]['dominant_race'])


from diffusers.utils.torch_utils import randn_tensor

generator = torch.manual_seed(42)

latents = nn.Parameter(randn_tensor((4, 4, 64, 64), generator=generator, device=pipe._execution_device, dtype=pipe.text_encoder.dtype))
latents0 = latents.data[:1].clone()
optimizer = torch.optim.SGD([latents], 1)  # 1 or 2



# prompt = 'A person and a person in the snow'
# prompt = 'a movie poster of a person and a person'
prompt = "Close-up photo of the happy smiles on the faces of the cool person and beautiful person as they leave the island with the treasure, sail back to the vacation beach, and begin their love story, 35mm photograph, film, professional, 4k, highly detailed."
# prompt = "A person and a person on the beach"
# prompt = "A person and a person sitting in a park"
# prompt = "A person and a person holding a bottle of red wine"
# prompt = "A person and a person standing together"
# prompt = "A person and a person riding a horse"

if attribute1[idx1]['dominant_gender'] == 'Man':
    prompt = prompt.replace('person', attribute1[idx1]['dominant_race'] + ' man', 1)
else:
    prompt = prompt.replace('person', attribute1[idx1]['dominant_race'] + ' woman', 1)

if attribute2[idx2]['dominant_gender'] == 'Man':
    prompt = prompt.replace('person', attribute2[idx2]['dominant_race'] + ' man', 1)
else:
    prompt = prompt.replace('person', attribute2[idx2]['dominant_race'] + ' woman', 1)
prompt = prompt
# prompt = prompt + ', faces'



import clip
from torchvision import transforms
from torchvision.transforms import functional as FT
from torchvision.transforms.functional import InterpolationMode   
def text2img_similarity(model, image, caption, device='cuda'):

    image = image.unsqueeze(0).to(device) 

    # image = F.interpolate(image, size=224, mode='bicubic', align_corners=False).squeeze(0)
    # _, h, w = image.shape
    # top = (h - 224) // 2
    # left = (w - 224) // 2
    # image = image[:, top:top + 224, left:left + 224]
    # image = image.unsqueeze(0)
    image = FT.resize(image, size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias='warn')
    image = FT.center_crop(image, (224, 224))
    image = image + torch.randn_like(image) * 0.005
    
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    image = (image - mean) / std


    text_tokens = clip.tokenize([caption]).to(device)

    logits_per_image, logits_per_text = model(image, text_tokens)

    return logits_per_image[0][0]

clip_model, clip_processor = clip.load("ViT-B/32", device='cuda')



from segment_anything import build_sam, SamPredictor
from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert

def person_detector(model,
                    image,
                    box_threshold,
                    text_threshold):
    image_source, image = load_image(image)

    detected_obj = []
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption='person',
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    h, w, _ = image_source.shape
    boxes = boxes * torch.tensor([w, h, w, h])
    boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    if len(boxes) > 0:
        boxes = [list(map(int, box)) for box in boxes]
        detected_obj.extend(boxes)
    return detected_obj

def process_image(image, device):
    image = image.unsqueeze(0).to(device) 
    
    image = FT.resize(image, size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias='warn')
    # image = kornia.geometry.transform.resize(image, (224, 224), interpolation='bicubic', align_corners=False)
    # print("after bubic",image)
    # print(image.shape)

    image = FT.center_crop(image, (224, 224))
    image = image + torch.randn_like(image) * 0.01
    # _, h, w = image.shape
    # top = (h - 224) // 2
    # left = (w - 224) // 2
    # image = image[:, top:top + 224, left:left + 224]
    # image = image.unsqueeze(0)
    # print("after center crop",image)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    image = (image - mean) / std
    return image

def char2char_similarity(dino_model,
                         clip_model,
                         sam_predictor,
                         processor,
                         generate_image_np,
                         generate_image,
                         character_image_list,
                         character_image_list_np,
                         device,
                         box_threshold=0.5,
                         text_threshold=0.25):
    output_similarity = 0
    person_boxes = person_detector(
        dino_model, generate_image_np, box_threshold, text_threshold)
    person_image_list = []
    for person_box in person_boxes:
        xmin, ymin, xmax, ymax = person_box
        person_image =  generate_image[:, ymin:ymax, xmin:xmax]
        person_image_list.append(person_image)
 
    character_similarity_list = []
    for character_image, character_image_np in zip(character_image_list, character_image_list_np):
        person_box = person_detector(
            dino_model, character_image_np, box_threshold, text_threshold)[0]
        xmin, ymin, xmax, ymax = person_box
        # print(person_box)
   
        character_image = character_image[:, ymin:ymax, xmin:xmax]
        # print("input",character_image)
        # print("input shape",character_image.shape)
        character_input = process_image(character_image, device)
        # print("output",character_input)
        # character_input = processor(character_image).unsqueeze(0).to(device)
        character_similarity = 0

        # with torch.no_grad():
        character_features = clip_model.encode_image(character_input)
        for person_image in person_image_list:
            # person_input = processor(person_image).unsqueeze(0).to(device)
            person_input = process_image(person_image, device)
            # print(person_input)
            person_features = clip_model.encode_image(person_input)
            cos = torch.nn.CosineSimilarity(dim=0)
            similarity = cos(character_features[0], person_features[0])
            if similarity > character_similarity:
                character_similarity = similarity
        character_similarity_list.append(character_similarity)
        output_similarity += character_similarity
    output_similarity = output_similarity / len(character_image_list)
    return output_similarity

dino_model = load_model(
            "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "./GroundingDINO/weights/groundingdino_swint_ogc.pth")
sam = build_sam(
    checkpoint='./models/sam/sam_vit_h_4b8939.pth')
sam_predictor = SamPredictor(sam)

character_image_list = [torch.tensor(np.array(ref_image1), dtype=torch.float32).permute(2, 0, 1).to(torch.device('cuda')) / 255.0, torch.tensor(np.array(ref_image2), dtype=torch.float32).permute(2, 0, 1).to(torch.device('cuda')) / 255.0]

character_image_list_np = [np.array(ref_image1), np.array(ref_image2)]



latents_last = latents.data.clone()
latents_last_e = latents.data.clone()
initialized_i = -1

def callback(self, i, t, callback_kwargs):
    global latents_last, latents_last_e, initialized_i
    if initialized_i < i:
        latents[i:(i+1)].data.copy_(callback_kwargs['latents'])
        latents_last[i:(i+1)].copy_(callback_kwargs['latents'])
        latents_last_e[i:(i+1)].copy_(callback_kwargs['latents'])
        initialized_i = i
    if i < 3:
        callback_kwargs['latents'] += latents[(i+1):(i+2)] - latents[(i+1):(i+2)].detach()
    latents_e = callback_kwargs['latents'].data.clone()
    callback_kwargs['latents'] += latents_last[i:(i+1)].detach() - callback_kwargs['latents'].detach()
    callback_kwargs['latents'] += latents_e.detach() - latents_last_e[i:(i+1)].detach()
    # callback_kwargs['latents'] += latents[i:(i+1)].detach() - latents_last_e[i:(i+1)].detach()
    callback_kwargs['latents'] += (latents[i:(i+1)].detach() - latents_last_e[i:(i+1)].detach()) * 0.95796674
    latents_last[i:(i+1)].copy_(callback_kwargs['latents'])
    latents_last_e[i:(i+1)].data.copy_(latents_e)
    latents[i:(i+1)].data.copy_(latents_e)
    return callback_kwargs

for epoch in tqdm(range(61)):
    t0 = time.time()


    image = my_forward(pipe, prompt=prompt, num_inference_steps=4, guidance_scale=3.0, latents=latents0+latents[:1]-latents[:1].detach(), output_type='pt', return_dict=False, callback_on_step_end=callback)[0][0]

    
    t1 = time.time()

    det_thresh_backup = detector.det_thresh
    boxes = []
    while len(boxes) <= 1:
        boxes, kpss = detector.detect(np.array(image.permute(1, 2, 0).detach().cpu().numpy() * 255, dtype=np.uint8), max_num=2)
        detector.det_thresh -= 0.1
    det_thresh_backup2 = detector.det_thresh + 0.1
    detector.det_thresh = det_thresh_backup
    t2 = time.time()

    M1 = insightface.utils.face_align.estimate_norm(kpss[0])
    image_cropped_1 = kornia.geometry.transform.warp_affine(
        image.float().unsqueeze(0), torch.tensor(M1).float().unsqueeze(0).to('cuda'), (112, 112)
    ) * 2 - 1
    M2 = insightface.utils.face_align.estimate_norm(kpss[1])
    image_cropped_2 = kornia.geometry.transform.warp_affine(
        image.float().unsqueeze(0), torch.tensor(M2).float().unsqueeze(0).to('cuda'), (112, 112)
    ) * 2 - 1
    embedding_1 = model(image_cropped_1)
    embedding_2 = model(image_cropped_2)
    ref_embeddings = torch.cat([ref_embedding1, ref_embedding2, ref_embedding1, ref_embedding2])
    t2i_similarity = text2img_similarity(clip_model,
                                        image=image,  
                                        caption='Close-up photo of the happy smiles on the faces of the cool man and beautiful woman as they leave the island with the treasure, sail back to the vacation beach, and begin their love story, 35mm photograph, film, professional, 4k, highly detailed.')


    print("t2i_similarity:", t2i_similarity.item())
    image_np = np.array(image.permute(1, 2, 0).detach().cpu().numpy() * 255, dtype=np.uint8)
    character_similarity = char2char_similarity(dino_model,
                                            clip_model,
                                            sam_predictor,
                                            processor=clip_processor,
                                            generate_image_np=image_np,
                                            generate_image=image,
                                            character_image_list_np=character_image_list_np,
                                            character_image_list=character_image_list,
                                            device='cuda')
    print("character_similarity:", character_similarity.item())
    proposal_embeddings = torch.cat([embedding_1, embedding_2, embedding_2, embedding_1])
    sim = F.cosine_similarity(ref_embeddings, proposal_embeddings)
    print("face1 similarity:", max(sim[0], sim[3]).item())
    print("face2 similarity:", max(sim[1], sim[2]).item())

    # loss = (50 - t2i_similarity) * 10+ (2 - max(sim[0] + sim[1], sim[2] + sim[3])) * 100 + (100 - character_similarity * 100) * 10

    if epoch >= 30 and max(sim[0], sim[3]).item() > 0.9:
       loss = (50 - t2i_similarity) * 10+ (1 - max(sim[1], sim[2])) * 100 + (100 - character_similarity * 100) * 10 
    elif epoch >= 30 and max(sim[1], sim[2]).item() > 0.9:
       loss = (50 - t2i_similarity) * 10+ (1 - max(sim[0], sim[3])) * 100 + (100 - character_similarity * 100) * 10 
    else:
       loss = (50 - t2i_similarity) * 10+ (2 - max(sim[0] + sim[1], sim[2] + sim[3])) * 100 + (100 - character_similarity * 100) * 10
    t3 = time.time()

    optimizer.zero_grad()
    loss.backward()
    t4 = time.time()
    grad_norm = latents.grad.reshape(4, -1).norm(dim=-1)
    latents.grad /= grad_norm.reshape(4, 1, 1, 1).clamp(min=1)
    # latents.grad.clamp_(min=-2e-2, max=2e-2)  # optional for removing artifacts
    optimizer.step()
    t5 = time.time()

    if epoch % 10 == 0:
        print('loss:', loss.data.item())
        print('grad:', grad_norm.tolist())
        print('time:', t1-t0, t2-t1, '(%f)' % det_thresh_backup2, t3-t2, t4-t3, t5-t4)
        plt.imshow(np.array(image.permute(1, 2, 0).detach().cpu() * 255, dtype=np.uint8))
        plt.axis("off")
        plt.show()
    # Convert the tensor to a NumPy array and scale it to 0-255
    image_np = np.array(image.permute(1, 2, 0).detach().cpu() * 255, dtype=np.uint8)

    # Create a PIL image
    image_pil = Image.fromarray(image_np)

    if not os.path.exists('./images'):
        os.makedirs('./images')
    # Save the image
    image_pil.save('./images/image_%d.png' % epoch)