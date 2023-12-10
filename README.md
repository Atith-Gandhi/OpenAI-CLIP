## Install requirements:

1. pip install ofa
2. Install the Flicker-8k dataset from [here](https://www.kaggle.com/datasets/adityajn105/flickr8k/) and unzip the file to Datasets/Flicker-8k/ folder
3. Download Dynabert zip file from [here](https://drive.google.com/file/d/1pYApaDcse5QIB6lZagWO0uElAavFazpA/view) and put it in the pretrained/ folder 
4. Run pip install -r requirements.txt

## Train the network

1. Run - python main.py --sampling-function {sampling_function}  
(choices=['randomized_sampling', 'big_small_sampling','no_sampling','supernet_subnet_sampling', 'width_balanced_sampling','depth_balanced_sampling', 'randomized_and_supernet_subnet_sampling']

