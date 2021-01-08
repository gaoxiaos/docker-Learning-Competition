# Base Images
## 从天池基础镜像构建(from的base img 根据自己的需要更换，建议使用天池open list镜像链接：https://tianchi.aliyun.com/forum/postDetail?postId=67720)
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/mmdetection:pytorch1.4-cuda10.1-py3

##安装依赖包,pip包请在requirements.txt添加
#RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

## 把当前文件夹里的文件构建到镜像的/mmdetection目录下,并设置为默认工作目录
ADD demo.jpg /mmdetection
ADD faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth /mmdetection
ADD detector.py /mmdetection
ADD run.sh /mmdetection
WORKDIR /mmdetection

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
