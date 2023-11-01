# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import pathlib
import pandas as pd
import cv2

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

def building_train_partition_train(df, n_partitions):

    idx = list(df.index.values)
    partition_dic = {}
    p_keys = list()

    # Creation of partition names list
    for i in range(0, n_partitions):
        p_keys.append('p' + str(i))

    total_images = len(idx)
    images_by_partition = int(total_images /n_partitions)
    i = 0
    n = images_by_partition
    #print(total_images)
    #print (i, ':', n)

    # Building the partition diccionary
    for p in range(0, n_partitions):
        partition_dic[p_keys[p]] = idx[i : n]
        if p < (n_partitions -2):
            i = n
            n= n + images_by_partition
        else:
            i = n
            n = total_images+1

        #print ('p', p+1, '-', i, ':', n)

    return(partition_dic)


def change_picture_resolution(source_path, dest_path, width, files_list):
    '''

    '''

    for filename in files_list:
        #filename = filename + '.jpg'
        #print(filename)
        img = (cv2.imread(os.path.join(source_path, filename)))
        l, w = img.shape[:2]
        interp = cv2.INTER_AREA
        aspect = w/l
        scaled_width = width
        scaled_long = int(scaled_width * aspect)

        scaled_img = cv2.resize(img, (scaled_long, scaled_width), interpolation=interp)
        cv2.imwrite(os.path.join(dest_path, filename), scaled_img)
        #print(f'{l,w, w/l, scaled_long, scaled_width}')


#
if __name__ == '__main__':
    # Constants
    original_path = 'C:\\Users\\jgonzalezleal\\Documents\\Uoc\\Data\\ISIC_2019_Training_Input'
    original_path_test = 'C:\\Users\\jgonzalezleal\\Documents\\Uoc\\Data\\ISIC_2019_Test_Input'
    converted_path = 'C:\\Users\\jgonzalezleal\\Documents\\Uoc\\Data\\Train'
    converted_path_test = 'C:\\Users\\jgonzalezleal\\Documents\\Uoc\\Data\\Test'
    prefix_file = 'ISIC_'
    images_list_path = 'C:\\Users\\jgonzalezleal\\Documents\\Uoc\\Data'
    images_list_filename = 'ISIC_2019_Training_GroundTruth.csv'
    test_image = 'ISIC_0000004.jpg'
    width = 200
    long = 200
    num_part = 10

    print('BEGIN PROCESS')
    print('-----------')

    # Extract the picture file name from file
    images_list_df = load_csv_to_df(os.path.join(images_list_path, images_list_filename))
    #print(images_list_df.head())

    # Diccionary with picture names into partitions (which are the dicc keys)
    dic= building_train_partition_train(images_list_df, 10)

    # Change the picture resolution and store into partition into Train folders
    for p in dic.keys():
        print(os.path.join(converted_path, p))
        change_picture_resolution(original_path,
                                  os.path.join(converted_path, p),
                                  width,
                                  dic[p]
                                  )


    # looking for the test files list
    n, df_test = num_files(original_path_test, prefix_file)
    print(f'Test file number: {n}')
    #print(df_test)

    # Change the picture resolution and store into Test folder
    change_picture_resolution(original_path_test,
                              converted_path_test,
                              width,
                              df_test
                              )
                              



    print('PROCESS FINISHED')
    print('-----------')
