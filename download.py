import clip
import torch
from PIL import Image
from torchvision import transforms
from segment_anything import build_sam, SamPredictor
from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert
clip_model, clip_processor = clip.load("ViT-B/32", device='cuda')

def person_detector(model,
                    image_path,
                    box_threshold,
                    text_threshold):
    image_source, image = load_image(image_path)

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
 

def char2char_similarity(dino_model,
                         clip_model,
                         sam_predictor,
                         processor,
                         generate_image,
                         character_image_list,
                         device,
                         box_threshold=0.5,
                         text_threshold=0.25):
    output_similarity = 0
    generate_image = generate_image
    person_boxes = person_detector(
        dino_model, generate_image, box_threshold, text_threshold)
    person_image_list = []
    for person_box in person_boxes:
        person_image = generate_image.crop(tuple(person_box))
        person_image_list.append(person_image)
 
    character_similarity_list = []
    for character_image in character_image_list:
        person_box = person_detector(
            dino_model, character_image, box_threshold, text_threshold)[0]
        character_image = character_image.crop(tuple(person_box))
        character_input = processor(character_image).unsqueeze(0).to(device)
        character_similarity = 0
    
        character_features = clip_model.encode_image(character_input)
        for person_image in person_image_list:
            person_input = processor(person_image).unsqueeze(0).to(device)

            person_features = clip_model.encode_image(person_input)
            cos = torch.nn.CosineSimilarity(dim=0)
            similarity = cos(character_features[0], person_features[0]).item()
            if similarity > character_similarity:
                character_similarity = similarity
        character_similarity_list.append(character_similarity)
        output_similarity += character_similarity
    output_similarity = output_similarity / len(character_image_list)
    return output_similarity

dino_model = load_model(
            "/root/wuyujia_proj/RectifID/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "/root/wuyujia_proj/RectifID/GroundingDINO/weights/groundingdino_swint_ogc.pth")
sam = build_sam(
    checkpoint='/root/wuyujia_proj/RectifID/models/sam/sam_vit_h_4b8939.pth')
sam_predictor = SamPredictor(sam)


character_similarity = char2char_similarity(dino_model,
                                        clip_model,
                                        sam_predictor,
                                        processor=clip_processor,
                                        generate_image_path=generate_image,
                                        character_image_path_list=character_image_list,
                                        device='cuda')
 