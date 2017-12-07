# Image Similarity using Deep Ranking

This repository is based on [this](https://medium.com/@akarshzingade/image-similarity-using-deep-ranking-c1bd83855978) blog. Please refer it to understand how Deep Ranking works.  

## ImageDataGeneratorCustom.py

For this implementation of Deep Ranking model, we need a image generator that would read from a text file which contains the triplets and then generate the images from those. The ImageDataGeneratorCustom.py contains the code that does this.

## DeepRanking.py

The Deep Ranking model consists of a ConvNet, 2 parallel small networks and a ranking loss function.

The 'convnet_model_' function contains the code for the ConvNet function. Currently, VGG16 has been used. You can modify this function to use any another network of your preference. 

The 'deep_rank_model' function creates the model for the whole Deep Ranking networks- combining the ConvNet and the 2 parallel small networks.

The 'loss_tensor' function implements the ranking loss described in the paper. You can change the value of the gap parameter 'g' in this function. Currently, it is set at 1. 

**Note**: The batch_size should always be a multiple of 3. Since the triplet images are passed sequentially, the batch_size has to be multiple of 3. 

## How to pass the path of Triplet text file?

The Image Generator class has been modified to take the path of the triplet text file as an input. So, to the Image Generator's flow_from_directory, you will pass the path of the triplet text file to 'triplet_file' variable.

Here's a snippet:
```javascript
class DataGenerator(object):
    def __init__(self, params, target_size=(224, 224)):
        self.params = params
        self.target_size = target_size
        self.idg = ImageDataGeneratorCustom(**params)

    def get_train_generator(self, batch_size):
        return self.idg.flow_from_directory("./dataset/",
                                            batch_size=batch_size,
                                            target_size=self.target_size,shuffle=False,
                                            triplet_path  ='./triplet_5033.txt'
                                           )

    def get_test_generator(self, batch_size):
        return self.idg.flow_from_directory("./dataset/",
                                            batch_size=batch_size,
                                            target_size=self.target_size, shuffle=False,
                                            triplet_path  ='./triplet_5033.txt'
                                        )
```

**Note**: The shuffle parameter should be set to False.

## tripletSampler.py

This is the triplet sampling method that I have implemented as I have described in my [blog](https://medium.com/@akarshzingade/image-similarity-using-deep-ranking-c1bd83855978). 

It takes 4 inputs- Input directory where the images are present, Output directory where the generated triplet text file is to be stored, the number of positive images for each query image, and number of negative images for each query image. 

Here's how to pass these information to the script:

```
Format: python triplet_sampler.py --input_directory <<path to the directory>> --output_directory <<path to the directory>> --num_pos_images <<Number of positive images you want>> --num_neg_images <<Number of negative images you want>>```

```Example: python triplet_sampler.py --input_directory similarity_images --output_directory triplet_folder --num_pos_images 10 --num_neg_images 50```
