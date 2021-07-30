import os
import cv2 as cv
import shutil as sh

# Dataset Utils - List Directory & Read
# x = os.listdir("./images")
# for i in x:
#     ix = cv.imread(os.path.join("./images", i))
#     cv.imshow("ListDir",ix)
#     cv.waitKey(delay=0)
# print(x)
# print(images_fps)

# x = os.listdir(r"C:\Users\Null\Downloads\archive\dataset_6\dataset_6")
# for i in range(len(x)):
#     files = x[i]
#     fp = r"C:\Users\Null\Downloads\archive\dataset_6\dataset_6"
#     fp += "\\" + files
#     dstp1 = r"C:\Users\Null\Desktop\Internship\MRP\root\MedSeg1\2\train\imgs"
#     dstp2 = r"C:\Users\Null\Desktop\Internship\MRP\root\MedSeg1\2\train\masks"
#     if files.startswith("s"):
#         sh.move(fp, dstp2)
#     print("Moved file - " + files + " Prog:"+str(i)+"/"+str(len(x)))
f=r"C:\Users\Null\Desktop\Internship\MRP\root\MedSeg1\2\test\masks\\"
x = os.listdir(r"C:\Users\Null\Desktop\Internship\MRP\root\MedSeg1\2\test\masks")
for i in range(len(x)):
    files = x[i]
    filex= files
    filex = filex.replace("segmentation", "volume")
    filex = filex.replace("lesionmask_", "s")
    filex = filex.replace("livermask_", "")
    os.rename(f+files,f+filex)
    print("Renamed file - " + files + " Prog:" + str(i) + "/" + str(len(x)))

