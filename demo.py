from face_recognition import FaceRecognition
import cv2

app = FaceRecognition()
app.prepare(ctx_id=0, det_size=(640, 640))
# img = ins_get_image('t1')

app.read_facebase("./img/targets")
img = cv2.imread("./IMG_0067.jpeg")
img = cv2.imread("./test.jpg")
rimg = app.draw_on(img)
cv2.imwrite("./test_output.jpg", rimg)