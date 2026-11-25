# Image Captioning with VGG16 and LSTM

Generates a text caption for an uploaded image. VGG16 extracts image features and an LSTM decoder writes the caption word by word using beam search. Includes a small Flask web app.

## Overview
- Dataset: Flickr8k (about 8,000 images with captions). Not included in this repo.
- Model: VGG16 image features + LSTM decoder, trained for 30 epochs with a checkpoint on validation loss.
- Training and experiments: ImageCaption.ipynb (built in Google Colab).
- Results: val_loss = [ADD], val_acc = [ADD]
- Web app (app.py): upload a JPG or PNG and get a caption. Uses OpenCV for image loading and beam search with beam width 5.

## Tech stack
Python, TensorFlow/Keras, OpenCV, NumPy, Flask

## Run locally
1. pip install -r requirements.txt
2. Run the notebook to train the model. It saves best_image_caption_model.keras, tokenizer.pkl and max_length.pkl. The model file is not in the repo because of its size.
3. python app.py
4. Open http://127.0.0.1:5000

## Team
Two-person project
## Future scope
Try attention, larger datasets like Flickr30k, and BLEU score evaluation.
