import albumentations as albu

def _aug_img_lbl(img_np, lbl_np, img_name=None):
    aug_results = []

    # img augmentation
    tf_res = albu.Compose([
        albu.HorizontalFlip(p=0.8), #垂直翻转
        albu.Flip(p=0.8), #水平翻转
        albu.Transpose(p=0.8), #转置
        albu.RandomScale(scale_limit=0.1,interpolation=1,always_apply=False,p=0.5), #随机缩放
        albu.RandomRotate90(p=0.8),#随机旋转90度
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=None,
                                      always_apply=False, p=0.8)#随机亮度对比度变化
    ],p=1.0)(image = img_np, mask = lbl_np)


    aug_results.append({"image": tf_res["image"],"mask": tf_res["mask"]})
    return aug_results