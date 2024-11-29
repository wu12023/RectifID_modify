from PIL import Image

def compare_images(image1_path, image2_path):
    # 打开图片
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # 获取图片尺寸
    width, height = image1.size

    # 逐个像素比较
    for x in range(width):
        for y in range(height):
            pixel1 = image1.getpixel((x, y))
            pixel2 = image2.getpixel((x, y))

            if pixel1!= pixel2:
                print(f"像素差异：({x}, {y})")

# 替换为你的图片路径
image1_path = '/data/hdd2/liangqian/RectifID/images/image_20_0.png'
image2_path = '/data/hdd2/liangqian/RectifID/images/image_20_1.png'

compare_images(image1_path, image2_path)