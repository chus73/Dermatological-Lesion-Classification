# This is a sample Python script.
import json
# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import pathlib
import random
import shutil

import pandas as pd
import cv2
from PIL import Image

def num_files(path, prefix_file):
    df = []
    initial_count = 0
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            if file.startswith(prefix_file):
                initial_count += 1
                df.append(file)
    #print(f'# files: {initial_count}')
    #print(f'# ficheros por partición: {int(initial_count/10)}')
    return (initial_count, df)

# Load csv file and store into df

def load_csv_to_df(file_name):
    '''
    file upload from disk.
    Input:
        - file_name: File name with extension file.
    Output:
        - A Pandas dataframe or a call to create a new dataframe
    '''
    try:
        df =  pd.read_csv(file_name, encoding='utf-8', index_col=0)
        return(df)
    except FileNotFoundError:
        print('No File was found')
        return(df)



def building_train_partition_train(df, n_partitions, validation_perc, seed, verbose=False):

    random.seed(seed)

    # random the name list
    idx = list(df.index.values)
    random.shuffle(idx)

    train_dict = {}
    test_dict = {}
    p_keys = list()

    # Creation of partition names list
    for i in range(0, n_partitions):
        p_keys.append('p' + str(i))

    # Calculate the images number by partition
    total_images = len(idx)
    images_by_partition = int(total_images /n_partitions)

    # Calculate the test images number by partition
    num_pictures_val = int(images_by_partition * validation_perc)

    # images_by_partition is equal to num_train + num_validation
    i = 0
    n = images_by_partition

    # Building the partition dictionary
    for p in range(0, n_partitions):
        n_train = n - num_pictures_val
        train_dict[p_keys[p]] = idx[i: n_train]
        test_dict[p_keys[p]] = idx[n_train: n]
        if verbose:
            print ('p', p, '-', i, ':', n)
        if p < (n_partitions -2):
            i = n
            n= n + images_by_partition
        else:
            i = n
            n = total_images+1
    return(train_dict, test_dict)


def resolution_images(image_path):

    files = os.listdir(image_path)
    im_resolution = []

    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            full_path = os.path.join(image_path, file)
            image = Image.open(full_path)
            width, high = image.size
            resolution = (width, high)
            try:
                im_resolution.index(resolution)
            except:
                im_resolution.append(resolution)
            #print(f"Resolución de {file}: {width}x{high}")
    return(im_resolution)


def change_picture_resolution(source_path, dest_path, width_resizing, width, long, files_list):
    '''

    '''

    print('')
    print(f'Source: {source_path}')
    print(f'Destination: {dest_path}')
    #print(files_list)

    for filename in files_list:

        # Image load process
        if os.path.splitext(filename)[1] == '':
            filename = filename + '.jpg'
        #print(filename)
        img = (cv2.imread(os.path.join(source_path, filename)))

        # Image reduction process
        l, w = img.shape[:2]
        interp = cv2.INTER_AREA
        aspect = w/l
        scaled_width = width_resizing
        scaled_long = int(scaled_width * aspect)
        scaled_img = cv2.resize(img, (scaled_long, scaled_width), interpolation=interp)

        # Padding process
        h, w, _ = scaled_img.shape
        top = (long - h) // 2
        bottom = long - (top + h)
        left = (width - w) // 2
        right = width - (left + w)
        # print(long, width)
        # print(h, w)
        # print(top, bottom, left, right)

        #padding_img = cv2.copyMakeBorder(scaled_img,top, bottom, left, right,cv2.BORDER_CONSTANT, value=[0, 0, 0])
        padding_img = cv2.copyMakeBorder(scaled_img, top, bottom, left, right, cv2.BORDER_REPLICATE)

        # Save the image to disc
        cv2.imwrite(os.path.join(dest_path, filename), padding_img)

    print('')



def return_label_from_image(images_list_df, picture_name):
    '''
    Looking for the picture label. The class is codify with a integer number.
    case stament has been introduce on 3.10 python version, but we don't use
    for compatibility reason.

    :param images_list_df:
    :param picture_name:
    :return:
    '''

    rec = images_list_df.loc[picture_name]

    label = ''
    if rec['MEL'] == 1:
        label = 1
    elif rec['NV'] == 1:
        label = 2
    elif rec['BCC'] == 1:
        label = 3
    elif rec['AK'] == 1:
        label = 4
    elif rec['BKL'] == 1:
        label = 5
    elif rec['DF'] == 1:
        label = 6
    elif rec['VASC'] == 1:
        label = 7
    elif rec['SCC'] == 1:
        label = 8
    else:
        label = 0

    return (label)


def y_label_build(images_list_df, picture_list):
    '''

    '''

    label_list = []

    # Making the label list
    for p in picture_list:
        label_list.append(return_label_from_image(images_list_df, p))

    return (label_list)
def folder_structure(partition_path, n_partitions):
    if n_partitions>0:
        for p in range(n_partitions):
            path_train = os.path.join(partition_path, 'P' + str(p), 'Train')
            path_val = os.path.join(partition_path, 'P' + str(p), 'Validation')
            try:
                shutil.rmtree(os.path.join(partition_path, 'P' + str(p)))
            except Exception as error:
                print("An exception occurred:", error)
            finally:
                os.makedirs(path_train)
                os.makedirs(path_val)
    else:
        try:
            shutil.rmtree(partition_path)
        except Exception as error:
            print("An exception occurred:", error)
        finally:
            os.makedirs(partition_path)

# BEGIN PROCESS
if __name__ == '__main__':

    # Constants
    original_path = 'C:\\Users\\jgonzalezleal\\Documents\\Uoc\\Data\\ISIC_2019_Training_Input'
    original_path_test = 'C:\\Users\\jgonzalezleal\\Documents\\Uoc\\Data\\ISIC_2019_Test_Input'
    converted_path_train = 'C:\\Users\\jgonzalezleal\\Documents\\Uoc\\Data\\Train'
    converted_path_test = 'C:\\Users\\jgonzalezleal\\Documents\\Uoc\\Data\\Test'
    prefix_file = 'ISIC_'
    seed = 1234
    images_list_path = 'C:\\Users\\jgonzalezleal\\Documents\\Uoc\\Data'
    images_list_filename = 'ISIC_2019_Training_GroundTruth.csv'
    #test_image = 'ISIC_0000004.jpg'
    width_resizing = 120 # For resizing
    width = 220         # Final resolution x
    long = 220          # Final resolution y
    num_part = 10       # Partition number
    validation_perc = 0.1 # Extract the Validation pictures set from Train set

    # Variables
    images_train_dict = {}
    images_test_dict = {}
    y_train_dict = {}
    y_val_dict = {}
    y_test_df = []

    print('BEGIN PROCESS')
    print('-----------')

    # Create the folder structure
    print('Step 1. Creating the folder structure')
    folder_structure(converted_path_train, num_part)
    folder_structure(converted_path_test, 0)

    # Extract the picture file name from file
    print('Step 2. Loading the images index file')
    images_list_df = load_csv_to_df(os.path.join(images_list_path, images_list_filename))

    # Create the dictionary with picture names into partitions (which are the dict keys)
    print('Step 3. Creating train & test images dictionary')
    images_train_dict, images_val_dict = building_train_partition_train(images_list_df, 10, validation_perc, seed, False)
    #[print(k, ': ', len(v)) for k, v in  images_train_dict.items()]
    #print('')
    #[print(k, ': ', len(v)) for k, v in images_val_dict.items()]

    print('Step 4. Creating train & test labels dictionary')
    # Compose the dictionary with train labels following a partition classification
    for k, v in images_train_dict.items():
        y_train_dict[k] = y_label_build(images_list_df, v)
        #print(len(y_train_dict[k]))

    # Compose the dictionary with test labels following a partition classification
    for k, v in images_val_dict.items():
        y_val_dict[k] = y_label_build(images_list_df, v)
        #print(len(y_val_dict[k]))

    print('Step 5. Creating the resample train & validation images')
    # Change the picture resolution and store into partition into Train & Validation folders
    for p in images_train_dict.keys():
        # Train conversion
        change_picture_resolution(original_path,
                                  os.path.join(converted_path_train, p, 'Train'),
                                  width_resizing,
                                  width,
                                  long,
                                  images_train_dict[p]
                                  )
        # Validation conversion
        change_picture_resolution(original_path,
                                  os.path.join(converted_path_train, p, 'Validation'),
                                  width_resizing,
                                  width,
                                  long,
                                  images_val_dict[p]
                                  )

        #resolution = resolution_images(os.path.join(converted_path_train, p, 'Train'))
        #print(f'Picture resolutions: {resolution}')
    
    # Store the y_train & y_val dictionary intro main folder
    print('Step 6. Writing train & Validation dictionary into main folder')
    with open(os.path.join(converted_path_train, 'y_train_dict.json'), 'w') as fp:
        json.dump(y_train_dict, fp)

    with open(os.path.join(converted_path_train, 'y_val_dict.json'), 'w') as fp:
        json.dump(y_val_dict, fp)


    # looking for the test files list
    print('Step 7. Creating the resample test images')
    n, df_test = num_files(original_path_test, prefix_file)
    print(f'Test files number: {n}')
    #print(df_test)

    # Change the picture resolution and store into Test folder
    change_picture_resolution(original_path_test,
                              converted_path_test,
                              width_resizing,
                              width,
                              long,
                              df_test
                              )

    #y_test_df = y_label_build(images_list_df, df_test)

    #with open(os.path.join(converted_path_test, 'y_test_dict.json'), 'w') as fp:
    #    json.dump(y_test_df, fp)


    print('PROCESS FINISHED')
    print('-----------')
