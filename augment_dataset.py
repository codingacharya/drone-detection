import os
import cv2
import albumentations as A
import glob

# Input folders
IMG_DIR = "data/images"
LBL_DIR = "data/labels"

# Output folders
OUT_IMG_DIR = "data/augmented/images"
OUT_LBL_DIR = "data/augmented/labels"
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# Augmentation pipeline
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.RGBShift(p=0.3),
    A.RandomGamma(p=0.3),
    A.Blur(blur_limit=3, p=0.3),
    A.GaussNoise(var_limit=(5,25), p=0.3),
    A.Rotate(limit=20, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.4),
    A.Perspective(p=0.3),
],
bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])
)

COUNT = 20  # augmentations per image

# Load images
image_files = glob.glob(os.path.join(IMG_DIR, "*.jpg")) + \
              glob.glob(os.path.join(IMG_DIR, "*.png")) + \
              glob.glob(os.path.join(IMG_DIR, "*.jpeg"))

if len(image_files) == 0:
    print(f"❌ No images found in {IMG_DIR}")
    exit()

for img_path in image_files:
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)

    lbl_path = os.path.join(LBL_DIR, f"{name}.txt")
    if not os.path.exists(lbl_path):
        print(f"⚠️ Missing label for {filename}, skipping...")
        continue

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to read {img_path}, skipping...")
        continue

    # Load YOLO labels
    with open(lbl_path, "r") as f:
        bboxes, classes = [], []
        for line in f:
            c, x, y, w, h = line.strip().split()
            bboxes.append([float(x), float(y), float(w), float(h)])
            classes.append(int(c))

    # Apply augmentations
    for i in range(COUNT):
        augmented = transform(image=img, bboxes=bboxes, class_labels=classes)
        aug_img = augmented["image"]
        aug_boxes = augmented["bboxes"]
        aug_classes = augmented["class_labels"]

        # Save image
        out_img_path = os.path.join(OUT_IMG_DIR, f"{name}_aug_{i}.jpg")
        cv2.imwrite(out_img_path, aug_img)

        # Save labels
        out_lbl_path = os.path.join(OUT_LBL_DIR, f"{name}_aug_{i}.txt")
        with open(out_lbl_path, "w") as f:
            for cl, (x, y, w, h) in zip(aug_classes, aug_boxes):
                f.write(f"{cl} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

print(f"✅ Augmentation complete! Images saved in {OUT_IMG_DIR}")
