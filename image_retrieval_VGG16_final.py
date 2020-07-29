import os
import cv2
import time
import threading
from datetime import timedelta
from retrieval.create_thumb_images import create_thumb_images
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify, flash
from retrieval.retrieval_VGG16 import load_model, load_data, extract_feature, load_query_image, sort_img, extract_feature_query
from image_encrypt import *
from AES import *
import numpy as np
import torch
import pymysql
import shutil
import argparse
from io import BytesIO
from PIL import Image
import sys

def dynamic_modification_timer(mutex):
    time.sleep(3600)
    #time.sleep(60) #测试时使用一分钟
    mutex.acquire()
    flag = False
    global gallery_feature
    global image_paths
    '''
    增加
    直接读取并加入到特征集中；
    并将图片加密写入数据库
    '''
    base_path = './static/image_database/'

    # Prepare model. 加载预训练的model
    # #可删除部分
    # model = load_model(pretrained_model='./retrieval/models/net_best.pth', use_gpu=True)
    # print("Model load successfully!")
    # gallery_feature = np.load('./retrieval/models/gallery_feature.npy')  #ndarray
    # image_paths = np.load('./retrieval/models/image_paths.npy')
    # gallery_feature = torch.from_numpy(gallery_feature)
    # image_paths = image_paths.tolist()
    # #到这
    update_path = './static/update_images/'
    for dir_name in os.listdir(update_path):
        data_loader = load_data(data_path=update_path+dir_name+'/',
                                batch_size=1,
                                shuffle=False,
                                transform='default',
                                )
        increase_feature, increase_image_paths = extract_feature(model=model,
                                                                dataloaders=data_loader)  # torch.Size([59, 2048]) # tensor list
        # 加密并加入数据库(修改成AES)
        if len(increase_image_paths) != 0:
            flag = True
            print('添加成功')
            conn = dbinfo()
            conncur = conn.cursor()
            try:
                for i in range(0, len(increase_image_paths)):
                    path = increase_image_paths[i]
                    entend_path = path[path.rfind('/') + 1:]
                    fp = open(path, 'rb')
                    AES_data = aes.encrypt(fp.read())
                    #base64_data = mybase64.encode(fp.read())
                    fp.close()
                    try:
                        sql_insertimage = "replace into image_AES (file_name,image_value) VALUE (%s, %s) "
                        conncur.execute(sql_insertimage, (entend_path, AES_data))
                        seatdic = conncur.fetchall()
                        conn.commit()
                        conn.close()
                    except pymysql.Error as e:
                        print("Error %d %s" % (e.args[0], e.args[1]))
                        sys.exit(1)
                    new_path = base_path + entend_path
                    shutil.copyfile(path, new_path)
                    os.remove(path)
                    increase_image_paths[i] = new_path
            except IOError as e:
                print("Error %d %s" % (e.args[0], e.args[1]))
                sys.exit(1)

            # end_pos = path.rfind('/') - 1
            # start_pos = path.rfind('/', 0, end_pos)
            # dir_name = path[start_pos + 1:end_pos+1]
            gallery_feature = np.load('./retrieval/models/'+dir_name+'_feature.npy')
            gallery_feature = torch.from_numpy(gallery_feature)
            image_paths = np.load('./retrieval/models/'+dir_name+'_paths.npy')
            image_paths = image_paths.tolist()
            gallery_feature = torch.cat((gallery_feature, increase_feature), 0)
            image_paths.extend(increase_image_paths)
            # print(gallery_feature)
            # print(image_paths)
            # np.save('./retrieval/models/gallery_feature.npy', gallery_feature.numpy())
            # np.save('./retrieval/models/image_paths.npy', np.array(image_paths))
            if flag == True:
                np.save('./retrieval/models/'+dir_name+'_feature.npy', gallery_feature.numpy())
                np.save('./retrieval/models/'+dir_name+'_paths.npy', np.array(image_paths))
            else:
                print('未修改')

    '''
    删除
    在数据库中删除，然后用它的文件名在image_path中查找得到下标，删除向量即可；
    '''

    # Prepare model. 加载预训练的model
    # #可删除部分
    # model = load_model(pretrained_model='./retrieval/models/net_best.pth', use_gpu=True)
    # print("Model load successfully!")
    # gallery_feature = np.load('./retrieval/models/gallery_feature.npy')  #ndarray
    # image_paths = np.load('./retrieval/models/image_paths.npy')
    # gallery_feature = torch.from_numpy(gallery_feature)
    # image_paths = image_paths.tolist()
    # #到这
    delete_path = './static/delete_images/'
    for dir_name in os.listdir(delete_path):
        data_loader = load_data(data_path=delete_path + dir_name + '/',
                                batch_size=1,
                                shuffle=False,
                                transform='default',
                                )
        delete_feature, delete_image_paths = extract_feature(model=model,
                                                            dataloaders=data_loader)  # torch.Size([59, 2048]) # tensor list
        if len(delete_image_paths) != 0:
            print('删除成功')
            conn = dbinfo()
            conncur = conn.cursor()
            del_sql = 'delete from image_AES where file_name = %s'
            flag = True
            for i in range(0, len(delete_image_paths)):
                path = delete_image_paths[i]
                os.remove(path)
                delete_image_paths[i] = path[path.rfind('/') + 1:]
                # 删除数据库数据
            try:
                conncur.executemany(del_sql, delete_image_paths)
                conn.commit()
            except:
                conn.rollback()
            conn.close()
            del_list = []
            for i in range(0, len(image_paths)):
                path = image_paths[i]
                extend_path = path[path.rfind('/') + 1:]
                if extend_path in delete_image_paths:
                    del_list.append(i)
                    # os.remove(path)
            # end_pos = path.rfind('/') - 1
            # start_pos = path.rfind('/', 0, end_pos)
            # dir_name = path[start_pos + 1:end_pos + 1]
            gallery_feature = np.load('./retrieval/models/' + dir_name + '_feature.npy')
            gallery_feature = torch.from_numpy(gallery_feature)
            image_paths = np.load('./retrieval/models/' + dir_name + '_paths.npy')
            image_paths = image_paths.tolist()
            gallery_feature = gallery_feature.numpy()
            gallery_feature = np.delete(gallery_feature, del_list, axis=0)
            image_paths = np.delete(image_paths, del_list)
            gallery_feature = torch.from_numpy(gallery_feature)
            image_paths = image_paths.tolist()

        if flag == True:
            np.save('./retrieval/models/' + dir_name + '_feature.npy', gallery_feature.numpy())
            np.save('./retrieval/models/' + dir_name + '_paths.npy', np.array(image_paths))
        else:
            print('未修改')
    mutex.release()
    global thread
    # 重复构造定时器
    thread = threading.Thread(target=dynamic_modification_timer, args=(mutex, ))
    thread.start()


#parsing instrutions

# parser = argparse.ArgumentParser(description='Image Retrieval')
# parser.add_argument('--update', action='store_true', help='update database')
# args = parser.parse_args()

# if args.update:
    # Create thumb images.  创建缩略图

create_thumb_images(full_folder='./static/image_database/',
                    thumb_folder='./static/thumb_images/',
                    suffix='',
                    height=200,
                    del_former_thumb=True,
                    )



# Prepare model. 加载预训练的model
load_model_time = time.time()
model = load_model(pretrained_model=os.path.join('./DealNet/checkpoint', 'VGG16', 'VGG16_300epoch.t7'), use_gpu=True)
print(time.time()-load_model_time)
print("Model load successfully!")
local_dir = './static/image_database/'
# Extract database features.
# 在数据库图片不改变的情况下 选择是否保存特征向量 以节约时间
# if args.update:
    # Extract database features.
extract_feature_time = time.time()
for dir_name in os.listdir(local_dir):
    # Prepare data set.
    #print(dir_name)
    data_loader = load_data(data_path=local_dir + dir_name+'/',
                            batch_size=1,
                            shuffle=False,
                            transform='default',
                            )
    gallery_feature, image_paths = extract_feature(model=model, dataloaders=data_loader) # torch.Size([59, 2048])

    #print(gallery_feature)
    np.save('./retrieval/models/'+dir_name+'_feature.npy', gallery_feature.numpy())
    np.save('./retrieval/models/'+dir_name+'_paths.npy', np.array(image_paths))
print(time.time()-extract_feature_time)
print("extract_feature save successfully!")
# else:
#     gallery_feature = np.load('./retrieval/models/gallery_feature.npy')
#     gallery_feature = torch.from_numpy(gallery_feature)
#     image_paths = np.load('./retrieval/models/image_paths.npy')
#     image_paths = image_paths.tolist()


# mybase64 = MyBase64()
# s = "vwxrstuopq34567ABCDEFGHIJyz012PQRSTKLMNOZabcdUVWXYefghijklmn89+/"
# mybase64.__init__(s)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#先写入数据库
key = '1234567890123456'
aes = AesEncryption(key)# 声明加密算法对象
wrote_database_time = time.time()
try:
    conn = dbinfo()
    conncur = conn.cursor()
    for second_dir in os.listdir(local_dir):
        for root,dirs,files in os.walk(local_dir+second_dir):
            for file_name in files:
                #print(file_name)
                image_path = os.path.join(local_dir+second_dir+'/',file_name)
                #print(image_path)
                #imagename,_ = os.path.splitext(filepath)
                # print(file_name)
                fp = open(image_path,'rb')
                AES_data = aes.encrypt(fp.read())
                #base64_data = mybase64.encode(fp.read())
                fp.close()
                try:
                    sql_insertimage="replace into image_AES (file_name,image_value) values (%s, %s) "
                    conncur.execute(sql_insertimage, (file_name, AES_data))
                    seatdic= conncur.fetchall()
                    conn.commit()
                except pymysql.Error as e :
                    print("Error %d %s" % (e.args[0],e.args[1]))
                    sys.exit(1)
                os.remove(image_path)
            os.rmdir(local_dir+second_dir)# 删除图片文件夹
    conn.close()
except IOError as e:
    print("Error %d %s" % (e.args[0],e.args[1]))
    sys.exit(1)
print(time.time()-wrote_database_time)
print('wrote database successfully')


#定时调度
mutex = threading.Lock()
thread = threading.Thread(target=dynamic_modification_timer, args=(mutex, ))
thread.start()

# Picture extension supported.
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg', 'JPEG'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# Set static file cache expiration time
# app.send_file_max_age_default = timedelta(seconds=1)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


@app.route('/', methods=['POST', 'GET'])  # add route
def image_retrieval():
    basepath = os.path.dirname(__file__)  # current path
    upload_path = os.path.join(basepath, 'static/upload_image', 'query.jpg')

    if request.method == 'POST':
        if request.form['submit'] == 'upload':
            if len(request.files) == 0:
                return render_template('upload_finish.html', message='Please select a picture file!',
                                       img_query='./static/upload_image/query.jpg?123456')
            else:
                f = request.files['picture']

                if not (f and allowed_file(f.filename)):
                    # return jsonify({"error": 1001, "msg": "Examine picture extension, only png, PNG, jpg, JPG, or bmp supported."})
                    return render_template('upload_finish.html',
                                           message='Examine picture extension, png、PNG、jpg、JPG、bmp support.',
                                           img_query='./static/upload_image/query.jpg')
                else:

                    f.save(upload_path)

                    # transform image format and name with opencv.
                    # img = cv2.imread(upload_path)   # 从原来的读取img
                    # cv2.imwrite(os.path.join(basepath, 'static/upload_image', 'query.jpg'), img) # 保存到 当前目录下

                    return render_template('upload_finish.html', message='Upload successfully!',
                                           img_query='./static/upload_image/query.jpg?123456')  # 点了upload之后的成功界面

        elif request.form['submit'] == 'retrieval':
            start_time = time.time()
            # Query.
            query_image = load_query_image('./static/upload_image/query.jpg')
            # Extract query features.
            query_feature, predicted = extract_feature_query(model=model, img=query_image)  # [1,10]
            print(classes[predicted[0][0]])
            gallery_feature = np.load('./retrieval/models/'+classes[predicted[0][0]]+'_feature.npy')
            gallery_feature = torch.from_numpy(gallery_feature)
            image_paths = np.load('./retrieval/models/'+classes[predicted[0][0]]+'_paths.npy')
            image_paths = image_paths.tolist()
            increase_feature = np.load('./retrieval/models/'+classes[predicted[0][1]]+'_feature.npy')
            increase_feature = torch.from_numpy(increase_feature)
            increase_image_paths = np.load('./retrieval/models/'+classes[predicted[0][1]]+'_paths.npy')
            increase_image_paths = increase_image_paths.tolist()
            gallery_feature = torch.cat((gallery_feature, increase_feature), 0)
            image_paths.extend(increase_image_paths)

            similarity, index = sort_img(query_feature, gallery_feature)
            sorted_paths = [image_paths[i] for i in index]
            #数据库查询之后，将查询到的图片解密存入特定文件夹，只取前9
            save_path = './static/temporary_file/'
            read_database_time = time.time()
            try:
                conn = dbinfo()
                conncur = conn.cursor()
                tmb_images = []
                for i in range(0, 10):
                    file_name = os.path.split(sorted_paths[i])[1]
                    #print(file_name)
                    sql_image = "select image_value from image_AES where file_name = %s"
                    conncur.execute(sql_image, (file_name,))
                    feed_back = conncur.fetchall()
                    img_value = feed_back[0]['image_value']
                    img_value.encode('utf-8').decode('gbk')
                    # decode()
                    byte_data = aes.decrypt(img_value)
                    #print(byte_data)
                    #byte_data = mybase64.decode(img_value)
                    image_data = BytesIO(byte_data)
                    #print(image_data)
                    img = Image.open(image_data)
                    tmb_images.append(save_path + file_name)
                    rgb_im = img.convert('RGB')
                    rgb_im.save(save_path+file_name)
                conn.commit()
                conn.close()

            except pymysql.Error as e:
                print(e)
                sys.exit(1)
            print(time.time()-read_database_time)

            # print(sorted_paths) # 打印出查找之后根据相似度进行排序后的图片路径
            # tmb_images = ['./static/temporary_file/' + os.path.split(sorted_path)[1] for sorted_path in sorted_paths]
            # sorted_files = [os.path.split(sorted_path)[1] for sorted_path in sorted_paths]

            return render_template('retrieval.html',
                                   message="Retrieval finished, cost {:3f} seconds.".format(time.time() - start_time),
                                   sml1=similarity[0], sml2=similarity[1], sml3=similarity[2], sml4=similarity[3],
                                   sml5=similarity[4], sml6=similarity[5], sml7=similarity[6], sml8=similarity[7],
                                   sml9=similarity[8],
                                   img1_tmb=tmb_images[0], img2_tmb=tmb_images[1], img3_tmb=tmb_images[2],
                                   img4_tmb=tmb_images[3], img5_tmb=tmb_images[4], img6_tmb=tmb_images[5],
                                   img7_tmb=tmb_images[6], img8_tmb=tmb_images[7], img9_tmb=tmb_images[8],
                                   img_query='./static/upload_image/query.jpg?123456')
    return render_template('upload.html')



if __name__ == '__main__':
    # app.debug = True
    app.run(host='127.0.0.1', port=8080, debug=True, use_reloader=False)