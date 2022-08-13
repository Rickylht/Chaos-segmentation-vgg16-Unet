import os
import cv2

def extract_liver(dataset_dir):
    src_img_names = os.listdir(dataset_dir)
    if src_img_names[0] == 'Liver':
        src_img_names.remove('Liver')
    src_img_num = len(src_img_names)
    new_dir = os.path.join(dataset_dir, "Liver")
    for num in range(src_img_num):
        src_img_path = os.path.join(dataset_dir, src_img_names[num])
        src_img = cv2.imread(src_img_path)   # 0表示灰度图，默认参数为1（RGB图像）
        result = 0
        for i in range(src_img.shape[0]):
            for j in range(src_img.shape[1]):
                for k in range(src_img.shape[2]):
                    if 55 <= src_img.item(i, j, k) <= 70:
                        result = 1  # 表示有肝脏
                        src_img.itemset((i, j, k), 255)
                    else:
                        src_img.itemset((i, j, k), 0)
        if result == 1:
            new_path = os.path.join(new_dir, src_img_names[num])
            cv2.imwrite(new_path, src_img)


if __name__ == '__main__':
    train_dir = os.path.join("data", "train", "ground")
    test_dir = os.path.join("data", "val", "ground")
    extract_liver(train_dir)
    extract_liver(test_dir)