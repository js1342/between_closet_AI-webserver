from django.shortcuts import render
from django.conf import settings

import os
import urllib.request
import json
import numpy as np
# import pycocotools.mask as mask
import cv2
from glob import glob
from PIL import Image
from collections import OrderedDict
from pycocotools import mask
import time

# S3용
import logging
import boto3
from botocore.exceptions import ClientError

# color_classification
import tensorflow as tf
from tensorflow.keras.models import load_model
import os.path as pth
import pandas as pd
from tqdm import tqdm
from numba import cuda

# Create your views here.
def get_image(request):
    return render(request, 'server/get_image.html', {})

# s3 버킷 업로드 함수
def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3', aws_access_key_id='AKIAX24HGBAX4YJD32VN', aws_secret_access_key= '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')
    
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def by_POST(request):
     if request.method == 'POST':
        # 현재경로 확인 -> /home/lab05/webserver/server/views.py
        # print(os.path.abspath(__file__))
        # 경로 이동 -> /home/lab05/data/cloth/
        os.chdir('/home/lab05/data/cloth/')

        # image_url은 통신으로 들어올 예정. 추후 request.POST 방식으로 image 읽어들이기
        image_url = request.POST.get('url')

        # 이미지 요청 및 다운로드
        urllib.request.urlretrieve(image_url, "test.jpg")

#         os.system('curl ' + image_url + " > test.jpg")
#         time.sleep(1)
        
        # 경로 이동 -> /home/lab05/
        os.chdir('/home/lab05/')
        
        ## json 생성 작업
        image_paths = glob(os.path.join('data', 'cloth','*.jpg'))
        image_paths.sort()
        #json 형식으로 담을 dict 생성
        file_data = OrderedDict()
        file_data['images'] = []

        # for문으로 mmdetection에서 사용할 file_data 형식으로 변형
        # 여기서 id 값은 추후 고유 id 값을 집어넣어줘야 함
        for i in range(len(image_paths)):
            img = Image.open(image_paths[i], mode='r')
            filename = image_paths[i].split('/')[-1]
            dic = {"file_name": filename, "height": img.size[0], "width": img.size[1], "id": np.random.randint(30)}
            file_data['images'].append(dic)
            file_data['categories'] = [{"id": 1, "name": "top", "supercategory": "\uc0c1\uc758"}, 
                                       {"id": 2, "name": "blouse", "supercategory": "\uc0c1\uc758"}, 
                                       {"id": 3, "name": "t-shirt", "supercategory": "\uc0c1\uc758"}, 
                                       {"id": 4, "name": "Knitted fabri", "supercategory": "\uc0c1\uc758"}, 
                                       {"id": 5, "name": "shirt", "supercategory": "\uc0c1\uc758"}, 
#                                        {"id": 6, "name": "bra top", "supercategory": "\uc0c1\uc758"}, 
                                       {"id": 6, "name": "hood", "supercategory": "\uc0c1\uc758"}, 
                                       {"id": 7, "name": "blue jeans", "supercategory": "\ud558\uc758"}, 
                                       {"id": 8, "name": "pants", "supercategory": "\ud558\uc758"}, 
                                       {"id": 9, "name": "skirt", "supercategory": "\ud558\uc758"}, 
#                                        {"id": 11, "name": "leggings", "supercategory": "\ud558\uc758"}, 
#                                        {"id": 12, "name": "jogger pants", "supercategory": "\ud558\uc758"}, 
                                       {"id": 10, "name": "coat", "supercategory": "\uc544\uc6b0\ud130"}, 
                                       {"id": 11, "name": "jacket", "supercategory": "\uc544\uc6b0\ud130"}, 
                                       {"id": 12, "name": "jumper", "supercategory": "\uc544\uc6b0\ud130"}, 
#                                        {"id": 16, "name": "padding jacket", "supercategory": "\uc544\uc6b0\ud130"}, 
#                                        {"id": 17, "name": "best", "supercategory": "\uc544\uc6b0\ud130"}, 
#                                        {"id": 18, "name": "kadigan", "supercategory": "\uc544\uc6b0\ud130"}, 
#                                        {"id": 19, "name": "zip up", "supercategory": "\uc544\uc6b0\ud130"}, 
                                       {"id": 13, "name": "dress", "supercategory": "\uc6d0\ud53c\uc2a4"},] 
#                                        {"id": 21, "name": "jumpsuit", "supercategory": "\uc6d0\ud53c\uc2a4"}]
        with open('./data/my_cloth_django.json', 'w', encoding='utf-8') as make_file:
            json.dump(file_data, make_file, ensure_ascii=False)
        
        ## command 실행으로 예측 -> 윤곽선 json 파일 생성
        crop_folder_path = '/home/lab05/mmdetection'
        os.chdir(crop_folder_path)
        os.system('python tools/test.py configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco_new.py work_dirs/mask_rcnn_x101_64x4d_fpn_1x_coco_new/epoch_2.pth --format-only --eval-options "jsonfile_prefix=./django_test"')
#         os.system('python tools/test.py configs/htc/htc_x101_64x4d_fpn_16x1_20e_coco_copy.py work_dirs/htc_x101_64x4d_fpn_16x1_20e_coco_copy/epoch_1.pth --format-only --eval-options "jsonfile_prefix=./django_test"')
#         os.system('python tools/test.py configs/htc/htc_x101_64x4d_fpn_16x1_20e_coco_copy.py work_dirs/htc_x101_64x4d_fpn_16x1_20e_coco_copy/epoch_1.pth --show-dir work_dirs/result')
        
    #         time.sleep(20)
        
        ## 윤곽선 따기
        file_path = "./django_test.segm.json"
        with open(file_path, 'r') as f:
            data = json.load(f)

        updated_data = []

        # For each annotation
        for annotation in data:

            # Initialize variables
            obj = {}
            segmentation = []
            segmentation_polygons = []
#             global mask
            # Decode the binary mask
            mask_list = mask.decode(annotation['segmentation'])
            mask_list = np.ascontiguousarray(mask_list, dtype=np.uint8)
            contours, hierarchy = cv2.findContours((mask_list).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Get the contours
            for contour in contours:
                contour = contour.flatten().tolist()
                segmentation.append(contour)
                if len(contour) > 4:
                    segmentation.append(contour)
            if len(segmentation) == 0:
                continue

            # Get the polygons as (x, y) coordinates
            for i, segment in enumerate(segmentation):
                poligon = []
                poligons = []
                for j in range(len(segment)):
                    poligon.append(segment[j])
                    if (j+1)%2 == 0:
                        poligons.append(poligon)
                        poligon = []
                segmentation_polygons.append(poligons)

            # Save the segmentation and polygons for the current annotation
            obj['segmentation'] = segmentation
            obj['segmentation_polygons'] = segmentation_polygons
            updated_data.append(obj)
        
        points_list = []
        for i in range(len(updated_data)):
            points_list.append(np.array(updated_data[i]['segmentation_polygons'][0]))
        
        img = cv2.imread('../data/cloth/'+filename)
        height = img.shape[0]
        width = img.shape[1]
        masks = np.zeros((height, width), dtype=np.uint8)
        
        black_cropped_image_name = []
        white_cropped_image_name = []
        for i in range(len(points_list)):
            points = points_list[i].reshape((1,) + points_list[i].shape)
            cv2.fillPoly(masks, points, (255))
            res = cv2.bitwise_and(img,img,mask = masks)
            rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
            cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
            
            # backgroud color -> white
            bg = np.ones_like(res, np.uint8)*255
            cv2.bitwise_not(bg,bg, mask=masks)
            res2 = bg+ res
            rect2 = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
            cropped2 = res2[rect2[1]: rect2[1] + rect2[3], rect2[0]: rect2[0] + rect2[2]]
            
            first = str(np.random.randint(10000))
            second = str(np.random.randint(10000))
            black = './cv2_test_' + first + '_' + second +'_black.jpg'
            white = './cv2_test_' + first + '_' + second +'_white.jpg'
            name_black = black.split('/')[-1]
            name_white = white.split('/')[-1]
            black_cropped_image_name.append(name_black)
            white_cropped_image_name.append(name_white)
            cv2.imwrite(black, cropped)
            cv2.imwrite(white, cropped2)
            cv2.waitKey(0)

#         white_image_path = glob(os.path.join('white','*.jpg'))
#         image_path = glob(os.path.join('*.jpg'))
        
#         white_cropped_image_name = []
#         for path in white_image_path:
#             f = path.split('/')[-1]
#             white_cropped_image_name.append(f)
        print(white_cropped_image_name)
        
#         cropped_image_name = []
#         for path in image_path:
#             cropped_image_name.append(path)
        print(black_cropped_image_name)

        # S3 Client 생성 후 cropped된 이미지들을 S3로 전송
        s3 = boto3.client('s3',aws_access_key_id=settings.AWS_ACCESS_KEY, aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)

        for image_to_s3 in black_cropped_image_name:
             filename = image_to_s3
             bucket_name = 'cropped-photo' 
             s3.upload_file(filename, bucket_name, filename)
        
        for images_to_s3 in white_cropped_image_name:
             filename = images_to_s3
             bucket_name = 'cropped-photo' 
             s3.upload_file(filename, bucket_name, filename)
        
        # 모델이 추론한 CategoryID 값을 리스트에 삽입
        filepath = './django_test.segm.json'
        with open(filepath) as json_file:
            json_data = json.load(json_file)
        if json_data == False :
            return render(request, 'server/value_check_page2.html', {'black_cropped_image_name':black_cropped_image_name})

        else :
            category_id = []
            for ii in range(len(json_data)):
                cloth_id = json_data[ii]['category_id']
                category_id.append(str(cloth_id))
            
            ### 색상 분류        
            file_list = []
            file_name_list = []

            for i in range(len(black_cropped_image_name)):
                file = black_cropped_image_name[i]
                filename = file.split('.')[0]
                file_list.append(file)
                file_name_list.append(filename)

                # 모든 이미지를 공통 사이즈로 변환
                original_image = cv2.imread('./' + file)
                resizeHeight = int(320)
                resizeWidth  = int(240)
                resized_image = cv2.resize(original_image, (resizeHeight, resizeWidth), interpolation = cv2.INTER_CUBIC)
                cv2.imwrite('./' + file, resized_image)
                cv2.waitKey(0)

            # 경로 설정
            path = './'

            # tfrecord 생성
            def _bytes_feature(value):
                """Returns a bytes_list from a string / byte."""
                if isinstance(value, type(tf.constant(0))):
                    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

            def _float_feature(value):
                """Returns a float_list from a float / double."""
                return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

            def _floatarray_feature(array):
                """Returns a float_list from a float / double."""
                return tf.train.Feature(float_list=tf.train.FloatList(value=array))

            def _int64_feature(value):
                """Returns an int64_list from a bool / enum / int / uint."""
                return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


            def _validate_text(text):
                """If text is not str or unicode, then try to convert it to str."""
                if isinstance(text, str):
                    return text
                elif isinstance(text, 'unicode'):
                    return text.encode('utf8', 'ignore')
                else:
                    return str(text)

            def to_test_tfrecords(id_list, test_id_list, tfrecords_name):
                print("Start converting")
                options = tf.io.TFRecordOptions(compression_type = 'GZIP')
                with tf.io.TFRecordWriter(path=pth.join(tfrecords_name+'.tfrecords'), options=options) as writer:
                    for id_, test_id in tqdm(zip(id_list, test_id_list), total=len(id_list), position=0, leave=True):
                        image_path = pth.join(path, id_)
                        _binary_image = tf.io.read_file(image_path)

                        string_set = tf.train.Example(features=tf.train.Features(feature={
                            'image_raw': _bytes_feature(_binary_image),
                            'id': _bytes_feature(test_id.encode()),
                        }))

                        writer.write(string_set.SerializeToString())    

            to_test_tfrecords(file_list,file_name_list, pth.join(path, 'tf_record_test'))

            test_tfrecord_path = path + '/tf_record_test.tfrecords'

            BATCH_SIZE = 128
            NUM_CLASS = 10
            img_size = (224,224) # <- 학습할때 썼던 이미지 사이즈 입력!

            image_feature_description_test = {
                'image_raw': tf.io.FixedLenFeature([], tf.string),
                'id': tf.io.FixedLenFeature([], tf.string),
            }


            def _parse_image_function_test(example_proto):
                return tf.io.parse_single_example(example_proto, image_feature_description_test)

            def map_func_test(target_record):
                img = target_record['image_raw']
                label = target_record['id']
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.dtypes.cast(img, tf.float32)
                return img, label

            def prep_func_test(image, label):
                result_image = image / 255
                result_image = tf.image.resize(result_image, img_size)

                return result_image, label

            test_dataset = tf.data.TFRecordDataset(test_tfrecord_path, compression_type='GZIP')
            test_dataset = test_dataset.map(_parse_image_function_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            test_dataset = test_dataset.map(map_func_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            test_dataset = test_dataset.batch(BATCH_SIZE)
            test_dataset = test_dataset.map(prep_func_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


            # 모델불러오기 및 예측
            model = load_model('../clothes_color_densenet201_20201218.h5', compile=False)
            pred = model.predict(test_dataset)
            color_list = np.argmax(pred,axis=1)

            color = []
            if type(color_list) == int:
                color.append(str(color_list))
            else:
                for number in color_list:
                    color.append(str(number))

            print("이 옷의 카테고리 아이디는 {}".format(category_id[0]))
            print("이 옷의 컬러 아이디는 {}".format(color[0]))
            # S3 업로드 후 해당 이미지 삭제 
            for remove_img in white_cropped_image_name:
                os.remove(crop_folder_path + '/' + remove_img)
                
            for remove_image in black_cropped_image_name:
                os.remove(crop_folder_path + '/' + remove_image)

            os.remove('./tf_record_test.tfrecords')

            # 추후 충돌 방지를 위해 django_test.segm.json 파일 삭제
#             os.remove('./django_test.segm.json')

            # GPU 끄기
            device = cuda.get_current_device()
            device.reset()

            return render(request, 'server/value_check_page.html', {'black_cropped_image_name':black_cropped_image_name, 'category_id':category_id, 'color':color, 'white_cropped_image_name':white_cropped_image_name})

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    