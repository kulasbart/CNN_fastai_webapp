from fastai.vision.all import *
import gradio as gr
import torch 
import fastai
import fastbook

__all__ = ['bear_type', 'learn', 'classify_image', 'categories', 'image', 'label', 'examples', 'intf']

def bear_type(x): return x[0].isupper()

learn = load_learner('export.pkl')

categories = ('black', 'grizzly', 'teddy')

def classify_bear(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = []

intf = gr.Interface(fn=classify_bear, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)