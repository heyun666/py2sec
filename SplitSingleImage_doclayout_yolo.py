import warnings
# 设置全局警告过滤器，忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

from doclayout_yolo import YOLOv10 # doclayout_yolo

import logging
# 设置日志级别为 ERROR，这样就不会输出 INFO, DEBUG, WARNING信息
logging.getLogger("doclayout_yolo").setLevel(logging.ERROR)

from PIL import Image # pillow
import numpy as np # numpy
import cv2 # opencv-python
import io

import torch # torch

_VERSION_ = '0.1.0'
_DATE_ = '2024-12-27'
_DONE_ = '''

调用大模型 doclayout_yolo，基本10s/page
'''


# # Open an image using PIL     //具有 RGB 通道的 HWC 格式。
# source = Image.open("image.jpg")
# # Create a random numpy array of HWC shape (640, 640, 3) with values in range [0, 255] and type uint8
# source = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype="uint8")
# # Read an image using OpenCV  //带有 BGR 频道的 HWC 格式 uint8 (0-255)
# source = cv2.imread("image.jpg")
# results = model([source])

class SplitImage:
    def __init__(self, img_PIL, model_path):
        self.model_path = model_path            # 模型路径

        self.SplitImgs = []                     # 存储切割后的图像对象，np.array/PIL/io.BytesIO，即插图
        self.splitRects = []                    # 存储切割后的图像的坐标元组，(x1, y1, x2, y2)，py程序调用可直接读取该变量
        self.img_result = None                  # 存储原图像切割、填充后的图像对象，即底图

        self.img_PIL = img_PIL                  # 原始图像对象，PIL格式
        self.img = None                         # 原始图像对象，cv2格式
        self.bin = None                         # 预处理后的图像对象，cv2格式

        if self.img_PIL.mode != 'RGB':
            self.img = np.array(self.img_PIL.convert('RGB'))
        else:
            self.img = np.array(self.img_PIL)
        # self.imagePreProcess()                  # 1. 图像预处理；灰度、二值化、反色；
        self.splitImage()                       # 2. 图像分割：y坐标分割，x坐标分割，再次y坐标分割，得到字符图像    


    def imagePreProcess(self):
        # 1\图像预处理：读取cv2格式，转为灰度图，二值化，颜色反转/白底变黑色
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin = (255 - thresh)
        self.bin = np.array(bin, dtype=np.uint8)

        # image_ = Image.fromarray(self.bin)
        # image_.show()

    # 20s/image; cls 类值: 3.figure, 4.figure-caption, 5.table, 6.table-caption
    # xyxy 左上右下坐标x1y1x2y2 [1043.0, 1749.0, 1741.0, 2256.0]
    # xywh 中心点+宽高          [1392.0, 2003.0, 698.0, 506.0]
    def splitImage(self):
        model = YOLOv10(self.model_path)
        # results = model([self.img])
        # for result in results:
        
        # Perform prediction
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        det_res = model.predict(
            self.img,           # Image to predict
            imgsz=640,          # Prediction image size // 1024 640 512 320
            conf=0.3,           # Confidence threshold // 0.2 0.3 0.4
            device=device       # Device to use (e.g., 'cuda:0' or 'cpu')
        )
        for result in det_res[0]:
            clses = result.boxes.cls
            for index, cls in enumerate(clses):
                if cls == 3:
                    x0, y0, x1, y1 = result.boxes.xyxy[index].round().tolist()
                    self.splitRects.append((int(x0), int(y0), int(x1-x0), int(y1-y0)))

    def imageConvert2PIL(self):
        if not self.splitRects:
            # print("No split rects found!")
            return None
        # 将OpenCV格式的图像数据转换为PIL图像
        img_result = self.img.copy()
        for i in range(len(self.splitRects)):
            (x0, y0, w, h) = self.splitRects[i]
            img_split = self.img[y0:y0+h, x0:x0+w]
            self.SplitImgs.append(Image.fromarray(img_split))

            img_result[y0:y0+h, x0:x0+w] = (255,255,255)    # 扣图，白色填充
        self.img_result = Image.fromarray(img_result)


    def imageConvert2BytesIO(self):
        if not self.splitRects:
            # print("No split rects found!")
            return None
        # 将OpenCV格式的图像数据转换为BytesIO格式的图像
        img_result_ = self.img_PIL.copy()                            # self.img.copy()
        
        # RGBA不能直接保存为JPEG，需要先转为RGB
        if self.img_PIL.mode == 'RGBA':
            img_crop = self.img_PIL.convert('RGB')
        else:
            img_crop = self.img_PIL.copy()
        
        for i in range(len(self.splitRects)):
            (x0, y0, w, h) = self.splitRects[i]
            img_split = img_crop.crop((x0, y0, x0+w, y0+h))     # PIL格式抠图 self.img_PIL

            # 调整图片大小
            # if img_split.width > 1000 or img_split.height > 1000:
            new_width = w #img_split.width // 2
            new_height = h #img_split.height // 2
            img_split = img_split.resize((new_width, new_height))

            img_bytesIO = io.BytesIO()
            img_split.save(img_bytesIO, format='JPEG', optimize=True, compress_level=9, dpi=(100,100), quality=50) #format='PNG', optimize=True)
            self.SplitImgs.append(img_bytesIO)

            if img_result_.mode not in ['1', 'L', 'RGBA', 'RGB']:
                img_result_ = img_result_.convert('RGB')
            if img_result_.mode == 'RGB': x = (255,255,255)
            elif img_result_.mode == 'RGBA': x = (255,255,255,255)     # 透明度为0，不透明
            elif img_result_.mode == 'L': x = (255)
            elif img_result_.mode == '1': x = 1
            img_white = Image.new(img_result_.mode, (w, h), x)
            img_result_.paste(img_white, (x0, y0, x0+w, y0+h))
            # img_result[y0:y0+h, x0:x0+w] = (255,255,255)            # 扣图，白色填充 //对cv2格式抠图
        # img_result = Image.fromarray(img_result)
        self.img_result = io.BytesIO()
        img_result_.save(self.img_result, format='PNG', optimize=True)


    def imageConvert2CV(self):
        if not self.splitRects:
            # print("No split rects found!")
            return None
        # OpenCV格式的图像存储
        img_result = self.img.copy()
        for i in range(len(self.splitRects)):
            (x0, y0, w, h) = self.splitRects[i]
            img_split = self.img[y0:y0+h, x0:x0+w]
            self.SplitImgs.append(img_split)

            img_result[y0:y0+h, x0:x0+w] = (255,255,255)    # 扣图，白色填充
        self.img_result = img_result


    def writePIL(self):
        if not self.SplitImgs:
            print("No SplitImgs found!")
            return None
        # 将切割后的PIL格式的图像保存为文件
        for i in range(len(self.SplitImgs)):
            self.SplitImgs[i].save('./' + f'xy_{i}.png')
        self.img_result.save('./' + f'xy_result.png')


    def writeBytesIO(self):
        if not self.SplitImgs:
            print("No SplitImgs found!")
            return None
        # 将切割后的BytesIO格式的图像保存为文件
        for i in range(len(self.SplitImgs)):
            with open('./' + f'xy_{i}.png', 'wb') as f:
                f.write(self.SplitImgs[i].getvalue())
        with open('./' + f'xy_result.png', 'wb') as f:
            f.write(self.img_result.getvalue())


    def writeCV2(self):
        if not self.SplitImgs:
            print("No SplitImgs found!")
            return None
        # 将切割后的cv2格式的图像保存为文件
        for i in range(len(self.SplitImgs)):
            cv2.imwrite('./' + f'xy_{i}.png', self.SplitImgs[i])
        cv2.imwrite('./' + f'xy_result.png', self.img_result)


# 调用示例：可以传入图片文件，PIL格式，np/cv2格式
if __name__ == '__main__':
    image_path = "000003.png"#"000003.png" "000016.png" "000018.png"
    # rects = splitImage(model_path, image_path)
    # print(rects)

    import time
    start_time = time.time()

    a = SplitImage(Image.open(image_path))
    
    # 转为PIL图像
    # a.imageConvert2PIL()
    # print(a.splitRects)
    # a.writePIL()

    # 转为BytesIO
    a.imageConvert2BytesIO()
    print(a.splitRects)
    a.writeBytesIO()

    # 转为cv2/numpy
    # a.imageConvert2CV()
    # print(a.splitRects)
    # a.writeCV2()
    # [(441, 217, 1121, 581), (315, 1667, 1396, 972)]

    end_time = time.time()
    print(end_time - start_time)
