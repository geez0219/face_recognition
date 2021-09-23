import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from insightface.utils import DEFAULT_MP_NAME
import os
import argparse

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

def run_video(app, source):
    cap = cv2.VideoCapture(source)
    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print(f"Error opening video stream or file from {source}")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        frame = app.draw_on(frame)
        if ret == True:
            # Display the resulting frame
            cv2.imshow('Frame',frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

def run_img(app, source, output):
    img = cv2.imread(source)
    img = app.draw_on(img)
    cv2.imwrite(output, img)

def main():
    parser = argparse.ArgumentParser(prog='face_regonition')
    subparsers = parser.add_subparsers(dest="mode", help='use which as input')

    # create the parser for the "webcam" command
    parser_webcam = subparsers.add_parser('webcam', help='use webcam')
    parser_webcam.add_argument('-f', '--facebase', type=str, help="path of image base", required=True)
    parser_webcam.add_argument('-d', '--device', type=int, help='webcam device index', default=0)

    # create the parser for the "video" command
    parser_video = subparsers.add_parser('video', help='use video')
    parser_video.add_argument('-f', '--facebase', type=str, help="path of image base", required=True)
    parser_video.add_argument('-i', '--input', type=str, help='path of input file', required=True)


    # create the parser for the "img" command
    parser_img = subparsers.add_parser('img', help='use img')
    parser_img.add_argument('-f', '--facebase', type=str, help="path of image base", required=True)
    parser_img.add_argument('-i', '--input', type=str, help='path of input file', required=True)
    parser_img.add_argument('-o', '--output', type=str, help='path of output file', required=True)


    args = parser.parse_args()


    app = FaceRecognition()
    app.prepare(ctx_id=0, det_size=(640, 640))
    app.read_facebase(args.facebase)

    if args.mode == "webcam":
        run_video(app, args.device)
    elif args.mode == "video":
        run_video(app, args.input)
    else: # argparse.mode == "img"
        run_img(app, args.input, args.output)

if __name__ == "__main__":
    main()