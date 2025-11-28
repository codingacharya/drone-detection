git clone https://github.com/codingacharya/drone-detection.git

cd drone-detection

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

pip install albumentations opencv-python

pip install ultralytics streamlit pillow numpy opencv-python albumentations

python augment_dataset.py

python train_model.py

