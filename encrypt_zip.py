import subprocess
import zipfile as zf
import os


class ZipObj():

    def __init__(self, filepath, zip_name, passwd):
        self.filepath = filepath
        self.zip_name = zip_name
        self.passwd = passwd

    def enCrypt(self, deleteSource=False):
        """
	        压缩加密，并删除原数据
            window系统调用rar程序

            linux等其他系统调用内置命令 zip -P123 tar source
            默认不删除原文件
        """
        target = self.zip_name + ".zip"
        source = self.filepath
        # cmd = ['rar', 'a', '-p%s' % (self.passwd.encode('ascii')), target, source]
        # print(self.passwd.encode('ascii'))
        cmd = "zip -P %s -r %s %s" % (self.passwd, target, source)
        # p = subprocess.Popen(cmd)
        p = subprocess.Popen(cmd, executable=r'C:\Program Files\WinRAR\WinRAR.exe')
        p.wait()
        if deleteSource:
            os.remove(source)

    def deCrypt(self):
        """
        使用之前先创造ZipObj类
        解压文件
        """
        zfile = zf.ZipFile(self.zip_name + ".zip")
        zfile.extractall(r"zipdata", pwd=self.passwd.encode('ascii'))


if __name__ == "__main__":
    zipo = ZipObj('./static/image_database/', './static/image_database', '123')
    zipo.enCrypt(deleteSource=False)
    # zipo.deCrypt()
