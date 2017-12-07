import glob
import json
import random
import csv
import os
import re
import argparse
import numpy as np

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


def get_negative_images(all_images,image_names,num_neg_images):
    random_numbers = np.arange(len(all_images))
    np.random.shuffle(random_numbers)
    if int(num_neg_images)>(len(all_images)-1):
        num_neg_images = len(all_images)-1
    neg_count = 0
    negative_images = []
    for random_number in list(random_numbers):
        if all_images[random_number] not in image_names:
            negative_images.append(all_images[random_number])
            neg_count += 1
            if neg_count>(int(num_neg_images)-1):
                break
    return negative_images

def get_positive_images(image_name,image_names,num_pos_images):
    random_numbers = np.arange(len(image_names))
    np.random.shuffle(random_numbers)
    if int(num_pos_images)>(len(image_names)-1):
        num_pos_images = len(image_names)-1
    pos_count = 0
    positive_images = []
    for random_number in list(random_numbers):
        if image_names[random_number]!= image_name:
            positive_images.append(image_names[random_number])
            pos_count += 1 
            if int(pos_count)>(int(num_pos_images)-1):
                break
    return positive_images

def triplet_sampler(directory_path, output_path,num_neg_images,num_pos_images):
    classes = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    all_images = []
    for class_ in classes:
        all_images += (list_pictures(os.path.join(directory_path,class_)))
    triplets = []
    for class_ in classes:
        image_names = list_pictures(os.path.join(directory_path,class_))
        for image_name in image_names:
            image_names_set = set(image_names)
            query_image = image_name
            positive_images = get_positive_images(image_name,image_names,num_pos_images)
            for positive_image in positive_images:
                negative_images = get_negative_images(all_images,set(image_names),num_neg_images)
                for negative_image in negative_images:
                    triplets.append(query_image+',')
                    triplets.append(positive_image+',')
                    triplets.append(negative_image+'\n')
            
    f = open(os.path.join(output_path,"triplets.txt"),'w')
    f.write("".join(triplets))
    f.close()


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--input_directory', 
                       help='A argument for input directory')

    parser.add_argument('--output_directory', 
                       help='A argument for output directory')

    parser.add_argument('--num_pos_images', 
                       help='A argument for the number of Positive images per Query image')

    parser.add_argument('--num_neg_images', 
                       help='A argument for the number of Negative images per Query image')                                              
    
    args = parser.parse_args()

    if args.input_directory == None:
        print('Input Directory path is required!')
        quit()
    elif args.output_directory == None:
        print('Output Directory path is required!')
        quit()
    elif args.num_pos_images == None:
        print('Number of Positive Images is required!')
        quit()
    elif args.num_neg_images == None:
        print('Number of Negative Images is required!')
        quit()
    elif int(args.num_neg_images) < 1:
        print('Number of Negative Images cannot be less than 1!')
    elif int(args.num_pos_images) < 1:
        print('Number of Positive Images cannot be less than 1!')

    if not os.path.exists(args.input_directory):
            print (args.input_directory+" path does not exist!")
            quit()

    if not os.path.exists(args.output_directory):
            print (args.input_directory+" path does not exist!")
            quit()

    print ("Input Directory: "+args.input_directory)
    print ("Output Directory: "+args.output_directory)
    print ("Number of Positive image per Query image: "+args.num_pos_images)
    print ("Number of Negative image per Query image: "+args.num_neg_images)

    triplet_sampler(directory_path=args.input_directory, output_path=args.output_directory, num_neg_images=args.num_neg_images, num_pos_images=args.num_pos_images)
