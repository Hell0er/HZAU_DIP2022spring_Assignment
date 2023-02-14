from google_images_download import google_images_download
import os


# 爬取图片
response = google_images_download.googleimagesdownload()
arguments = {"keywords":"whaet+septoria","limit":4000,"print_urls":True,"chromedriver":"D:\\NotOnlyCode\\DIP\\chromedriver.exe"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)  

# 重命名
path_name='D:/NotOnlyCode/DIP/downloads/test/test'
i=0
for item in os.listdir(path_name):#进入到文件夹内，对每个文件进行循环遍历
    os.rename(os.path.join(path_name,item),os.path.join(path_name,(str(i)+'.jpg')))#os.path.join(path_name,item)表示找到每个文件的绝对路径并进行拼接操作
    i+=1
