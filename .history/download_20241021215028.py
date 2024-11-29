import clip
import torch
from PIL import Image
from torchvision import transforms
clip_model, clip_processor = clip.load("ViT-B/32", device='cuda')
def text2img_similarity(model,
                        processor,
                        image_path,
                        caption,
                        device='cuda'):
    generation_image = Image.open(image_path).convert('RGB')
# 定义转换操作
    # 定义转换器
    to_tensor = transforms.ToTensor()

    print(to_tensor(generation_image))
    transform = transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),  
        transforms.CenterCrop(size=(224, 224)), 
        transforms.Lambda(lambda img: img.convert("RGB")),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                            std=(0.26862954, 0.26130258, 0.27577711)) 
    ])
    image = transform(generation_image).unsqueeze(0).to(device)
    image2 = processor(generation_image).unsqueeze(0).to(device)
 
    text = clip.tokenize([caption]).to(device)
 
    with torch.no_grad():
 
        logits_per_image, logits_per_text = model(image2, text)
        probs = logits_per_image.cpu().numpy().tolist()
    return probs[0][0]



print(text2img_similarity(clip_model, clip_processor, './images/image_50.png', 'Close-up photo of the happy smiles on the faces of the cool man and beautiful woman as they leave the island with the treasure, sail back to the vacation beach, and begin their love story, 35mm photograph, film, professional, 4k, highly detailed.'))