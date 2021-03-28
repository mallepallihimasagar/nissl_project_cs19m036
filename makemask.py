import urllib.request
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tqdm import tqdm

# Adding information about user agent
opener=urllib.request.build_opener()
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)
def make_dataset(json_file_path):
    count=1
    with open(json_file_path) as f:
        annotations = json.load(f)
    with tqdm(annotations) as tdm:    
        for data in tdm:
            tdm.set_description('Downloading images')
            tdm.set_postfix(images_downloaded=f'{count}/{len(annotations)} images')
            count+=1
            try:
                os.mkdir(data['External ID'][:-4])
            except:
                continue

            # setting filename and image URL
            filename = data['External ID'][:-4]+'/'+data['External ID']
            image_url = data['Labeled Data']

            # calling urlretrieve function to get resource
            urllib.request.urlretrieve(image_url, filename)
            file,handle= urllib.request.urlretrieve(image_url)

            #creating mask for cell1 
            main_c =[]
            for l in data['Label']['cell1']:
                coordinates = l['geometry']
                small_c = []
                for c in coordinates:
                    x = c['x']
                    y = c['y']
                    small_c.append([x,y])
                main_c.append(small_c)

            mask1 = np.zeros((500,500),dtype=np.uint8)
            for small_c in main_c: 
                contours = np.array(small_c)
                cv2.fillPoly(mask1, pts =[contours], color=(255,255,255))
            mask1_filename = data['External ID'][:-4]+'/'+data['External ID'][:-4]+'_mask1.png'
            cv2.imwrite(mask1_filename,mask1)

            #creating mask for cell2
            main_c =[]
            for l in data['Label']['cell2']:
                coordinates = l['geometry']
                small_c = []
                for c in coordinates:
                    x = c['x']
                    y = c['y']
                    small_c.append([x,y])
                main_c.append(small_c)

            mask2 = np.zeros((500,500),dtype=np.uint8)
            for small_c in main_c: 
                contours = np.array(small_c)
                cv2.fillPoly(mask2, pts =[contours], color=(255,255,255))
            mask2_filename = data['External ID'][:-4]+'/'+data['External ID'][:-4]+'_mask2.png'
            cv2.imwrite(mask2_filename,mask2)

            #creating mask for cell3
            main_c =[]
            mask3 = np.zeros((500,500),dtype=np.uint8)
            try:
                for l in data['Label']['cell3']:
                    coordinates = l['geometry']
                    small_c = []
                    for c in coordinates:
                        x = c['x']
                        y = c['y']
                        small_c.append([x,y])
                    main_c.append(small_c)


                for small_c in main_c: 
                    contours = np.array(small_c)
                    cv2.fillPoly(mask3, pts =[contours], color=(255,255,255))
                mask3_filename = data['External ID'][:-4]+'/'+data['External ID'][:-4]+'_mask3.png'
                cv2.imwrite(mask3_filename,mask3)
            except:
                mask3_filename = data['External ID'][:-4]+'/'+data['External ID'][:-4]+'_mask3.png'
                cv2.imwrite(mask3_filename,mask3)
            
        
    print(f'Dataset created for {len(annotations)} images')
    
make_dataset('annotations/29_labelbox.json') 
    
