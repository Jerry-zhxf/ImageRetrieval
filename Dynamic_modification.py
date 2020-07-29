import shutil
from ImageRetrieval.retrieval.retrieval_Resnet50 import load_model, load_data, extract_feature
import numpy as np
import torch
import os
from ImageRetrieval.image_encrypt import *
flag = False
'''
增加
直接读取并加入到特征集中；
并将图片加密写入数据库
'''
base_path = './static/image_database/'
data_loader = load_data(data_path='./static/update_images/',
                        batch_size=1,
                        shuffle=False,
                        transform='default',
                        )
# Prepare model. 加载预训练的model
# #可删除部分
# model = load_model(pretrained_model='./retrieval/models/net_best.pth', use_gpu=True)
# print("Model load successfully!")
# gallery_feature = np.load('./retrieval/models/gallery_feature.npy')  #ndarray
# image_paths = np.load('./retrieval/models/image_paths.npy')
# gallery_feature = torch.from_numpy(gallery_feature)
# image_paths = image_paths.tolist()
# #到这
increase_feature, increase_image_paths = extract_feature(model=model, dataloaders=data_loader) # torch.Size([59, 2048]) # tensor list
#加密并加入数据库
mybase64 = MyBase64()
s = "vwxrstuopq34567ABCDEFGHIJyz012PQRSTKLMNOZabcdUVWXYefghijklmn89+/"
mybase64.__init__(s)
if len(increase_image_paths) != 0:
    flag = True
    try:
        for path in increase_image_paths:
            entend_path = path[path.rfind('/') + 1:]
            fp = open(path,'rb')
            base64_data = mybase64.encode(fp.read())
            fp.close()
            try:
                conn=dbinfo()
                conncur = conn.cursor()
                sql_insertimage="insert into image_base64 (file_name,image_value) VALUE (%s, %s) "
                conncur.execute(sql_insertimage, (entend_path,base64_data))
                seatdic= conncur.fetchall()
                conn.commit()
                conncur.close()
                conn.close()
            except pymysql.Error as e :
                print("Error %d %s" % (e.args[0],e.args[1]))
                sys.exit(1)
            new_path = base_path+entend_path
            shutil.copyfile(path, new_path)
            os.remove(path)
            path = new_path
    except IOError as e:
        print("Error %d %s" % (e.args[0],e.args[1]))
        sys.exit(1)

    gallery_feature = torch.cat((gallery_feature, increase_feature), 0)
    image_paths.extend(increase_image_paths)
    # print(gallery_feature)
    # print(image_paths)
    # np.save('./retrieval/models/gallery_feature.npy', gallery_feature.numpy())
    # np.save('./retrieval/models/image_paths.npy', np.array(image_paths))


'''
删除
在数据库中删除，然后用它的文件名在image_path中查找得到下标，删除向量即可；
'''
data_loader = load_data(data_path='./static/delete_images/',
                        batch_size=1,
                        shuffle=False,
                        transform='default',
                        )
# Prepare model. 加载预训练的model
# #可删除部分
# model = load_model(pretrained_model='./retrieval/models/net_best.pth', use_gpu=True)
# print("Model load successfully!")
# gallery_feature = np.load('./retrieval/models/gallery_feature.npy')  #ndarray
# image_paths = np.load('./retrieval/models/image_paths.npy')
# gallery_feature = torch.from_numpy(gallery_feature)
# image_paths = image_paths.tolist()
# #到这
delete_feature, delete_image_paths = extract_feature(model=model, dataloaders=data_loader) # torch.Size([59, 2048]) # tensor list
if len(delete_image_paths) != 0:
    flag = True
    for i in range(0, len(delete_image_paths)):
        path = delete_image_paths[i]
        os.remove(path)
        delete_image_paths[i] = path[path.rfind('/') + 1:]
    # 删除数据库数据
    conn = dbinfo()
    conncur = conn.cursor()
    sql = 'delete from image_base64 where file_name = %s'
    try:
        conncur.executemany(sql, delete_image_paths)
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
            os.remove(path)

    gallery_feature = gallery_feature.numpy()
    gallery_feature = np.delete(gallery_feature, del_list, axis=0)
    image_paths = np.delete(image_paths, del_list)
    gallery_feature = torch.from_numpy(gallery_feature)
    image_paths = image_paths.tolist()

if flag == True:
    np.save('./retrieval/models/gallery_feature.npy', gallery_feature.numpy())
    np.save('./retrieval/models/image_paths.npy', np.array(image_paths))
