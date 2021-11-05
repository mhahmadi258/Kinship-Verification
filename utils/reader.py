import os
import numpy as np
from scipy import io
import cv2
import h5py


# This method get dateset location and use
# meta data file in that location to generate 
# a general h5 file, which contains images vector 
# as nummpy array, image pairs label and theirs fold number.
def read_data(path):
    files = ['fd_pairs','fs_pairs','md_pairs','ms_pairs']
    dirs = ['father-dau','father-son','mother-dau','mother-son']

    maper = {'fd_pairs':0,'fs_pairs':1,'md_pairs':2,'ms_pairs':3}

    p_img_list = list()
    c_img_list = list()
    general_meta_data = None
    rel_type_list = list()

    for meta_file_name,image_dir_name in zip(files,dirs):
        meta_path = os.path.join(os.path.join(path,'meta_data',meta_file_name))
        images_path = os.path.join(os.path.join(path,'images'),image_dir_name)

        meta_data = io.loadmat(meta_path)['pairs']
        meta_data = np.array([[i[0][0][0],i[1][0][0],i[2][0],i[3][0]] for i in meta_data])
        general_meta_data = np.vstack((general_meta_data,meta_data[:,:2])) if general_meta_data is not None else meta_data[:,:2]
        
        for row in meta_data:
            parent_image = cv2.imread(os.path.join(images_path,row[2]))
            parent_image = cv2.resize(parent_image,dsize=(224,224))
            p_img_list.append(parent_image)

            child_image = cv2.imread(os.path.join(images_path,row[3]))
            child_image = cv2.resize(child_image,dsize=(224,224))
            c_img_list.append(child_image)

            rel_type_list.append(maper[meta_file_name])

    
    p_imgs = np.array(p_img_list)
    c_imgs = np.array(c_img_list)
    rel_types = np.array(rel_type_list)

    hf = h5py.File('data.h5','w')
    hf.create_dataset('parents',data=p_imgs)
    hf.create_dataset('child',data=c_imgs)
    hf.create_dataset('rel_types',data=rel_types)
    hf.create_dataset('meta_data',data=general_meta_data.astype(int))
    hf.close()
