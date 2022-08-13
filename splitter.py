import os
import shutil
import random

class Spliter:
    def __init__(self):
        #定义train和val的路径
        self.train_img_dir = ".\\data\\train\\img"
        self.train_lbl_dir = ".\\data\\train\\ground"
        self.test_img_dir = ".\\data\\val\\img"
        self.test_lbl_dir = ".\\data\\val\\ground"

    def get_path(self,patient_dir):
        lbl_paths = []
        img_paths = []

        t1_img_dir = os.path.join(patient_dir, "T1DUAL")
        lbl_dir = os.path.join(t1_img_dir, "Ground")#label路径
        lbl_names = os.listdir(lbl_dir)
        nums_lbl = len(lbl_names)
        # 拼接Ground文件夹的文件，存入到lbl_paths列表中
        for i in range(nums_lbl):
            lbl_paths.append(os.path.join(lbl_dir, lbl_names[i]))

        img_dir = os.path.join(t1_img_dir, "DICOM_anon\\InPhase")
        img_names = os.listdir(img_dir)
        # 拼接img文件夹的文件，存入到img_paths列表中
        for i in range(len(img_names)):
            img_paths.append(os.path.join(img_dir, img_names[i]))

        return lbl_paths, img_paths
    
    def main(self):
        dataset_dir = os.path.join("CHAOS_Train_Sets", "Train_Sets", "MR")
        train_split_rate = 0.8
        val_split_rate = 0.2

        patient_number_list = [1,2,3,5,8,10,13,15,19,20,21,22,31,32,33,34,36,37,38,39]
        random.shuffle(patient_number_list)
        n = 0
        for i in range(len(patient_number_list)):
            if n <= int(len(patient_number_list) * train_split_rate):
                patient_dir = os.path.join(dataset_dir, str(patient_number_list[i]))
                lbl_paths, img_paths = self.get_path(patient_dir)
                for j in range(len(lbl_paths)):
                    new_lbl_path = os.path.join(self.train_lbl_dir, "T1_Patient%s_No%d.png" % (patient_number_list[i],j))
                    shutil.copy(lbl_paths[j], new_lbl_path)

                for j in range(len(img_paths)):
                    new_img_path = os.path.join(self.train_img_dir, "T1_Patient%s_No%d.dcm" % (patient_number_list[i], j))
                    shutil.copy(img_paths[j], new_img_path)
                n += 1
            
            else:
                patient_dir = os.path.join(dataset_dir, str(patient_number_list[i]))
                lbl_paths, img_paths = self.get_path(patient_dir)
                for j in range(len(lbl_paths)):
                    new_lbl_path = os.path.join(self.test_lbl_dir, "T1_Patient%s_No%d.png" % (patient_number_list[i],j))
                    shutil.copy(lbl_paths[j], new_lbl_path)

                for j in range(len(img_paths)):
                    new_img_path = os.path.join(self.test_img_dir, "T1_Patient%s_No%d.dcm" % (patient_number_list[i], j))
                    shutil.copy(img_paths[j], new_img_path)
                n += 1
                

if __name__ == '__main__':
    Spliter().main()
    