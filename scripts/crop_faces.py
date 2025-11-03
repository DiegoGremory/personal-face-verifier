import os
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

DEVICE = 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=DEVICE)

def crop_one(img_path):
    img = Image.open(img_path).convert('RGB')
    boxes, _ = mtcnn.detect(img)  # boxes is None or array (N,4)
    if boxes is None or len(boxes)==0:
        return None
    # elegir la caja más grande por área
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    i = int(max(range(len(areas)), key=lambda k: areas[k]))
    box = boxes[i].astype(int)
    cropped = img.crop((box[0], box[1], box[2], box[3])).resize((160,160))
    return cropped

def run(src_dir='data', dst_dir='data/cropped'):
    for label in ['me','not_me']:
        src = os.path.join(src_dir, label)
        dst_label = os.path.join(dst_dir, label)
        os.makedirs(dst_label, exist_ok=True)
        for fn in tqdm(os.listdir(src), desc=label):
            if not fn.lower().endswith(('.jpg','.jpeg','.png')): continue
            p = os.path.join(src, fn)
            cropped = crop_one(p)
            if cropped is None:
                # registrar imagen sin cara detectada
                continue
            cropped.save(os.path.join(dst_label, fn))

run()