# Image Similarity using Deep Ranking

This repository is based on [this](https://medium.com/@akarshzingade/image-similarity-using-deep-ranking-c1bd83855978) blog. Please refer it to understand how Deep Ranking works.  

## ImageDataGeneratorCustom.py

For this implementation of Deep Ranking model, we need a image generator that would read from a text file which contains the triplets and then generate the images from those. The ImageDataGeneratorCustom.py contains the code that does this.

## deepRanking.py

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
Format: python triplet_sampler.py --input_directory <<path to the directory>> --output_directory <<path to the directory>> --num_pos_images <<Number of positive images you want>> --num_neg_images <<Number of negative images you want>>
```

```
Example: python triplet_sampler.py --input_directory similarity_images --output_directory triplet_folder --num_pos_images 10 --num_neg_images 50
```

For your custom dataset, you will have to segregate similar images to different folders. The structure of your dataset will be as follows:

```
dataset/
|__SimilarityClass1/
|    |__Image1.jpg, Image2.jpg and so on....
|
|__SimilarityClass2/
|    |_Image1.jpg, Image2.jpg and so on....
|
and so on...
```
Then, run the tripletSampler.py script on this folder.

## How to get the similarity distance between two images?

The "deepranking_get_distance.py" calculates the similarity distance between the two images and prints it out. The script takes 3 inputs:
1) The deepranking model file (h5 file)
2) Image 1
3) Image 2

```
Format: python deepranking_get_distance.py --model <<path to the model>> --image1 <<path to image 1>> --image2 <<path to image 2>>
```

```
Example: python deepranking_get_distance.py --model ./deepranking.h5 --image1 ./image1.jpg --image2 ./image2.jpg
```

## Ideas to compare your test image with your training data

For image search, you will have to compute the distance between the source image and all the images in the dataset. But, there are ways to optimise this search. One way is to

1) Cluster all the dataset image embeddings and store the cluster means.
2) Compare the source image embeddings with the cluster means.
3) Select the closest cluster.
4) Compare the source image embeddings with the dataset images belonging to the selected cluster.

You can also create hierarchical clusters to further optimise it.

## Known Issues

```
Issue: IndexError Out of Bounds.

Hacky Fix: Please ensure that the number of triplets in the triplets.txt is a multiple of (3 * batch_size).
```
