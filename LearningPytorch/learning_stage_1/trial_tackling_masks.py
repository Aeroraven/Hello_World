import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
mask_path=r"D:\liver2\liver2\test\masks\285.png"
image_path=r"D:\liver2\liver2\test\imgs\285.png"
class_values=[0]
image = cv.imread(mask_path)
image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
mask = cv.imread(mask_path,0)
t = (mask == 1)
masks = [(mask == v) for v in class_values]
mask2 = np.stack(masks, axis=-1).astype('float')
plt.imshow(mask2)
