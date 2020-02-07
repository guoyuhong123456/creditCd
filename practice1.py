# 导入工具包
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import myutils
# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

# 绘图函数

def cv_show(name, img1):
    cv2.imshow(name, img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 读取模板
img = cv2.imread("template.png")
cv_show('img', img)

# 转化为灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('ref', ref)

# 转化为二值图像
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('ref', ref)

# 计算轮廓
ref_, refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
cv_show('img', img)
print(np.array(refCnts).shape)
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# 遍历每一个轮廓
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 预处理
image = cv2.imread('object.png')
cv_show('image', image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray', gray)

# 礼貌操作，突出高亮
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat', tophat)
gradx = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradx = np.absolute(gradx)
(minVal, maxVal) = (np.min(gradx), np.max(gradx))
gradx = (255*((gradx-minVal)/(maxVal-minVal)))
gradx = gradx.astype("uint8")
print(np.array(gradx).shape)
cv_show('gradx', gradx)

# 闭操作
gradx = cv2.morphologyEx(gradx, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradx', gradx)
thresh = cv2.threshold(gradx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh', thresh)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv_show('thresh', thresh)

# 计算轮廓
thresh_, threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
cv_show('img', cur_img)
locs = []

# 遍历轮廓
for(i, c)in enumerate(cnts):
    # 计算矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w/float(h)
    if ar > 2.5 and ar < 4.0:
        if(w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x, y, w, h))
# 排序
locs = sorted(locs, key=lambda x: x[0])
output = []

# 遍历数字
for(i, (gx, gy, gw, gh)) in enumerate(locs):
    groupOutput = []
    group = gray[gy - 5:gy + gh + 5, gx - 5:gx + gw + 5]
    cv_show('group', group)

    # 预处理
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group', group)
    group_, digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

    # 计算每一组中的每一个数值
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        cv_show('roi', roi)
        scores = []

        for (digit, digitROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        groupOutput.append(str(np.argmax(scores)))
    cv2.rectangle(image, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gx, gy - 15), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.65, (0, 0, 255), 2)
    output.extend(groupOutput)

print("Credit Card Type:{}".format(FIRST_NUMBER[output[0]]))
print("Credit Card # : {}".format("".join(output)))
cv2.imshow("image", image)
cv2.waitKey(0)






