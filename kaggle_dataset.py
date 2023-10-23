import os
os.environ['KAGGLE_USERNAME'] = "chenghao13"
os.environ['KAGGLE_KEY'] = "eb7fe710afd05e285d3445a267bc231e" # Enter your Kaggle key here

# For Flickr 8k
os.system('kaggle datasets download -d adityajn105/flickr8k')
os.system('unzip flickr8k.zip')
dataset = "8k"