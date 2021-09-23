import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import pdb
import os

from numpy.lib.stride_tricks import as_strided
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
# img = ins_get_image('t1')

# img = cv2.imread("./test.jpg")
# faces = app.get(img)
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./test_output.jpg", rimg)

target_path = "./img/targets"
embedding_dict = {}

for name in os.listdir(target_path):
    tmp = []
    for file in os.scandir(os.path.join(target_path, name)):
        if file.is_file():
            img = cv2.imread(file.path)
            faces = app.get(img)
            if len(faces) > 1:
                bbox_area = []
                for face in faces:
                    bbox = face["bbox"]
                    bbox_area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                face = faces[np.argmax(bbox_area)]
            else:
                face = faces[0]

            tmp.append(face.normed_embedding)
    tmp = np.mean(tmp, axis=0)
    embedding_dict[name] = tmp / np.linalg.norm(tmp)

pdb.set_trace()