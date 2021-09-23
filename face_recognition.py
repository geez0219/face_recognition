import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from insightface.utils import DEFAULT_MP_NAME
import os

class FaceRecognition(FaceAnalysis):
    def get_single_face(self, img):
        faces = self.get(img)
        if len(faces) > 1:
            bbox_area = []
            for face in faces:
                bbox = face["bbox"]
                bbox_area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            result = faces[np.argmax(bbox_area)]

        elif len(faces) == 1:
            result = faces[0]

        else:
            result = None
        return result

    def read_facebase(self, path):
        self.facebase = {}
        for name in os.listdir(path):
            tmp = []
            for file in os.scandir(os.path.join(path, name)):
                if not file.is_file():
                    continue
                img = cv2.imread(file.path)
                face = self.get_single_face(img)
                if not face:
                    print(f"Warning: {file.path} image cannot detect any face!")
                else:
                    tmp.append(face.normed_embedding)
            tmp = np.mean(tmp, axis=0)
            self.facebase[name] = tmp / np.linalg.norm(tmp)

    @staticmethod
    def similarity(vec1, vec2):
        return np.dot(vec1, vec2)

    def classify(self, features, threshold):
        if not self.facebase:
            raise RuntimeError("Must use read_facebase() first before using self.classify()")

        for name, vec in self.facebase.items():
            if self.similarity(features, vec) > threshold:
                return name

        return "other"

    def classify_img(self, img, threshold):
        face = self.get_single_face(img)
        if not face:
            return None

        return self.classify(face.normed_embedding, threshold)


    def get_single_features(self, img):
        face = self.get_single_face(img)
        if not face:
            return None
        return face.normed_embedding

    def draw_on(self, img, threshold=0.5):
        faces = self.get(img)
        dimg = img.copy()
        color = (0, 0, 255)
        for face in faces:
            box = face.bbox.astype(np.int)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            text = self.classify(face.normed_embedding, threshold)
            cv2.putText(dimg,f'{text}', (box[0]-1, box[1]-4), cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

        return dimg