# _*_ coding: utf-8 _*_
#拓展工具遍历文件夹，对文件夹图片进行base64编码，写入数据库，并且读取查找方法
import random
import pymysql
import sys
from datetime import datetime
import base64
import os
from io import BytesIO
from PIL import Image


class MyBase64(object):
    STANDARD_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'  # standard base64 alphabet

    def __init__(self, alphabet=None):
        if alphabet == None:
            alphabet = MyBase64.STANDARD_ALPHABET
        if len(alphabet) != len(MyBase64.STANDARD_ALPHABET):
            raise RuntimeError('MyBase64 init error:alphabet len should equal 64')
        self.alphabet = alphabet

    def encode(self, data_to_encode):
        encoded = base64.b64encode(data_to_encode).decode()
        t = str.maketrans(MyBase64.STANDARD_ALPHABET, self.alphabet)
        return encoded.translate(t)

    def decode(self, string_to_decode):
        t = str.maketrans(self.alphabet, MyBase64.STANDARD_ALPHABET)
        encoded = string_to_decode.translate(t)
        return base64.b64decode(encoded)

    # @staticmethod
    # def random_alphabet():
    #     temp = MyBase64.STANDARD_ALPHABET
    #     out = ''
    #     while (True):
    #         size = len(temp)
    #         if size <= 0:
    #             break
    #         index = random.randint(0, size - 1)
    #         out = out + temp[index]
    #         if index + 1 >= size:
    #             temp = temp[0:index]
    #         else:
    #             temp = temp[0:index] + temp[index + 1:]
    #     return out

# mybase64 = MyBase64()
# s = "vwxrstuopq34567ABCDEFGHIJyz012PQRSTKLMNOZabcdUVWXYefghijklmn89+/"
# mybase64.__init__(s)
# print(mybase64.encode(b'1234'))
# print(mybase64.decode('5Epf6v=='))

def dbinfo():
    conn = ""
    conn = pymysql.connect(host='localhost', port=3306, user="root", password='root', database='image',
                           charset='utf8',cursorclass =pymysql.cursors.DictCursor)
    cur = conn.cursor()
    if not cur:
        return "access db is fail!"
    else:
        return conn

# local_dir = './image_database/'
# try:
#     for root,dirs,files in os.walk(local_dir):
#         for file_name in files:
#             image_path = os.path.join(local_dir,file_name)
#             #imagename,_ = os.path.splitext(filepath)
#             print(file_name)
#             fp = open(image_path,'rb')
#             base64_data = mybase64.encode(fp.read())
#             fp.close()
#             try:
#                 conn=dbinfo()
#                 conncur = conn.cursor()
#                 sql_insertimage="insert into image_base64 (file_name,image_value) VALUE (%s, %s) "
#                 conncur.execute(sql_insertimage, (file_name,base64_data))
#                 seatdic= conncur.fetchall()
#                 conn.commit()
#                 conn.close()
#
#             except pymysql.Error as e :
#                 print("Error %d %s" % (e.args[0],e.args[1]))
#                 sys.exit(1)
# except IOError as e:
#     print("Error %d %s" % (e.args[0],e.args[1]))
#     sys.exit(1)

# save_path = './encrypt/'
# try:
#     conn = dbinfo()
#     conncur = conn.cursor()
#     sql_selectimage = "select file_name, image_value from image_base64"
#     conncur.execute(sql_selectimage)
#     softpath = conncur.fetchall()
#     #print(softpath)
#     filepathlist = [x['file_name'].encode('utf-8').decode('gbk') for x in softpath]
#     imagepathlist = [x['image_value'].encode('utf-8').decode('gbk') for x in softpath]
#
#     for i in range(0, len(filepathlist)):
#         # decode()
#         byte_data = mybase64.decode(imagepathlist[i])
#         image_data = BytesIO(byte_data)
#         img = Image.open(image_data)
#         img.save(save_path+filepathlist[i])
#     conn.commit()
#     conn.close()
#
# except pymysql.Error as e :
#     print(e)
#     sys.exit(1)
