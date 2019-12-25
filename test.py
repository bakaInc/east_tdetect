import cv2 as cv


img = cv.imread('ig_2.jpg', 0)
#ret, thresh = cv.threshold(img, 70, 255, 0)
ret, thresh = cv.threshold(img, 240, 255, cv.THRESH_BINARY_INV)
thresh = cv.bitwise_not(thresh)

cv.imshow("Perfectlyfittedellipsethreshthreshthreshs", thresh)
cnts, hiers = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
#cv.waitKey(0)
img_s = img.copy()
img_s = cv.cvtColor(img_s, cv.COLOR_GRAY2RGB)
for cnt in cnts:
    if(len(cnt)>5):
      ellipse = cv.fitEllipse(cnt)
      cv.ellipse(img_s, ellipse, (255, 128, 0), 1, cv.LINE_AA)

cv.imshow("img", img)
cv.imshow("img_s", img_s)
cv.imshow("thresh", thresh)
cv.waitKey(0)
#cv.drawContours(img, contours, -1, (0, 255, 0), 2, maxLevel=0)
#cv.imshow("img", img)
#cv.waitKey(0)

#img_res = img.copy()

#ellipse = cv.fitEllipse(contours[0])
#img_res = cv.ellipse(img, ellipse, (0, 0, 255), thickness=3)

#cv.imshow("img_res", img_res)

#cv.waitKey(0)
#if len(contours) != 0:
#  for i in range(len(contours)):
#    if len(contours[i]) >= 5:
#      cv.drawContours(thresh,contours,-1,(150,10,255),3)
#      ellipse=cv.fitEllipse(contours[0])
    #else:
      # optional to "delete" the small contours
      #cv.drawContours(thresh,contours,-1,(0,0,0),-1)
#cv.imshow("Perfectlyfittedellipses",thresh)
cv.waitKey(0)