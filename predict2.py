# Import required packages
from imutils.contours import sort_contours
import cv2
import imutils
import numpy as np
import tensorflow as tf
# Read image from which text needs to be extracted
model = tf.keras.models.load_model('./Model/az_handwritten/3/az_model')
image = cv2.imread("Data/Test2/merapikan.jpg")

# Preprocessing the image starts

# Convert the image to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Performing OTSU threshold
ret, thresh1 = cv2.threshold(
	gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

# Creating a copy of image
contours = sort_contours(contours, method="left-to-right")[0]

# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file
height = image.shape[0]
chars = []
for cnt in contours:
	x, y, w, h = cv2.boundingRect(cnt)
	# extract the character and threshold it to make the character
	# appear as *white* (foreground) on a *black* background, then
	# grab the width and height of the thresholded image
	if (h >= int(height * 0.3)):
		roi = gray[y:y + h, x:x + w]
		# kernel = np.ones((5, 5), np.uint8)
		thresh = cv2.threshold(
			roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		# thresh = cv2.dilate(thresh, kernel, iterations=1)
		(tH, tW) = thresh.shape
		# if the width is greater than the height, resize along the
		# width dimension
		if tW > tH:
			thresh = imutils.resize(thresh, width=28)
		# otherwise, resize along the height
		else:
			thresh = imutils.resize(thresh, height=28)
		# re-grab the image dimensions (now that its been resized)
		# and then determine how much we need to pad the width and
		# height such that our image will be 28x28
		(tH, tW) = thresh.shape
		dX = int(max(0, 28 - tW) / 2.0)
		dY = int(max(0, 28 - tH) / 2.0)
		# pad the image and force 32x32 dimensions
		padded = cv2.copyMakeBorder(
			thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
		padded = cv2.resize(padded, (28, 28))
		# prepare the padded image for classification via our
		# handwriting OCR model
		padded = padded.astype("float32") / 255.0
		padded = np.expand_dims(padded, axis=-1)
		# update our list of characters that will be OCR'd
		chars.append((padded, (x, y, w, h)))

boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")


preds = model.predict(chars)
# define the list of label names
labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

for (pred, (x, y, w, h)) in zip(preds, boxes):
    # find the index of the label with the largest corresponding
    # probability, then extract the probability and label
    i = np.argmax(pred)
    prob = pred[i]
    label = labelNames[i]
    # draw the prediction on the image
    print("[INFO] {} - {:.2f}%".format(label, prob * 100))
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    # show the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
