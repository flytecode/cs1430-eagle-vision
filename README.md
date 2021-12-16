# Bird Species Image Classifier Web App


This is a simple image classification web app to predict the species depicted in an uploaded bird image.

The dataset used is the [CalTech-UCSD Birds 200](https://www.tensorflow.org/datasets/catalog/caltech_birds2011). This is a dataset with photos of 200 bird species, and 6033 images. We used the 2010 version.

<p align="center">
  <img width="700" height="500" src=http://www.vision.caltech.edu/visipedia/collage.jpg>
</p>

Below is the screenshot of the simple web app:

Special thanks to for the [tutorial](https://towardsdatascience.com/deploying-an-image-classification-web-app-with-python-3753c46bb79) on deploying an image classification web app with python.

## Running

First, install the requirements.txt using `pip install -r requirements.txt` within the working directory.

Then, to start the local server: `streamlit run app.py`

You should now be able to use the web app to upload your own images!

## Files

- Code directory: used to research and play with code
- colab.py: model code to generate a model file that could be loaded, the homework code did not generate a loadable model
- app.py: streamlite instructions to deploy model locally
- heroku files, however I was unable to deploy due to slug file being too large. (As a first-time heroku user, I wasn't sure how to fix this)
