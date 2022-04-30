# 图像拼接

# SIFT
# https://blog.csdn.net/abcjennifer/article/details/7639681?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165128289916782350918424%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165128289916782350918424&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-7639681.142^v9^pc_search_result_control_group,157^v4^control&utm_term=SIFT&spm=1018.2226.3001.4187
# RANSAC
# https://blog.csdn.net/robinhjwy/article/details/79174914?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165128564416782395337125%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165128564416782395337125&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-79174914.142^v9^pc_search_result_control_group,157^v4^control&utm_term=RANSAC&spm=1018.2226.3001.4187


import cv2
import numpy as np


def cv_show(name, img_r):
    cv2.imshow(name, img_r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像灰度处理
imageL = cv2.imread('L_503_0.jpg')
imageR = cv2.imread('R_503_0.jpg')
grayL = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)

# SIFT特征获取
sift = cv2.SIFT_create()
kpL = sift.detect(grayL, None)
kpL, desL = sift.compute(grayL, kpL)
kpR = sift.detect(grayR, None)
kpR, desR = sift.compute(grayR, kpR)
kpL = np.float64([kp.pt for kp in kpL])
kpR = np.float64([kp.pt for kp in kpR])

# KNN特征匹配 RANSAC视角转换
matcher = cv2.BFMatcher()
KNN_matches = matcher.knnMatch(desL, desR, 2)
matches = []
for m in KNN_matches:
    if len(m) == 2 and m[0].distance <= m[1].distance * 0.99:
        matches.append((m[0].trainIdx, m[0].queryIdx))
    if len(matches) > 4:
        ptsL = np.float32([kpL[i] for (_, i) in matches])
        ptsR = np.float32([kpR[i] for (i, _) in matches])
        (H, status) = cv2.findHomography(ptsR, ptsL, cv2.RANSAC)
if H is None:
    print(None)

# 两图拼接
print(H)
print(imageL.shape, imageR.shape)
result = cv2.warpPerspective(imageR, H, (imageL.shape[1] + imageR.shape[1], imageL.shape[0]))
print(result.shape)
cv_show('image_L', result)
result[0:imageR.shape[0], 0:imageL.shape[1]] = imageL
cv_show('xx', result)
cv2.imwrite('result.jpg', result)
