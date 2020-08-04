# ImageRetrieval
构建一个加密图像检索系统，通过使用深度学习算法对存储的图像进行分类和标记，并以此作为图 像检索的依据，并设计一个检索算法，进而提高加密图像检索的准确性和效率。
该算法主要运行在图像检索服务器端。框架：Pytorch

在cmd打python+文件名即可；

图像数据库默认在\ImageRetrieval\static\image_database，因为初始化之后，会被删除，因此每一次系统开始运行时需要先到\ImageRetrieval\static\back-up1将图像复制到image_database,在运行代码；
update_images和delete_images则是实现更新的文件夹，用于管理；


索引存放：\ImageRetrieval\retrieval\models
训练模型代码：\ImageRetrieval\DealNet
模型存放位置：\ImageRetrieval\DealNet\checkpoint

\ImageRetrieval\retrieval中的代码是实现特征提取和检索主要功能的核心代码，在外层需要调用到它们；

requirements.txt中说明了环境配置的各种版本；

image.sql中有数据库表的设计结构和数据；
