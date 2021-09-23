import argparse
import pdb
def main():
    parser = argparse.ArgumentParser(prog='face_regonition')
    subparsers = parser.add_subparsers(dest="mode", help='use which as input')

    # create the parser for the "a" command
    parser_webcam = subparsers.add_parser('webcam', help='use webcam')
    parser_webcam.add_argument('-f', '--facebase', type=str, help="path of image base", required=True)
    parser_webcam.add_argument('-d', '--device', type=int, help='webcam device index', default=0)

    # create the parser for the "b" command
    parser_video = subparsers.add_parser('video', help='use video')
    parser_video.add_argument('-f', '--facebase', type=str, help="path of image base", required=True)
    parser_video.add_argument('-i', '--input', type=str, help='path of input file', required=True)
    parser_video.add_argument('-o', '--output', type=str, help='path of input file', default=None)


    # create the parser for the "b" command
    parser_img = subparsers.add_parser('img', help='use img')
    parser_img.add_argument('-f', '--facebase', type=str, help="path of image base", required=True)
    parser_img.add_argument('-i', '--input', type=str, help='path of input file', required=True)
    parser_img.add_argument('-o', '--output', type=str, help='path of output file', required=True)


    args = parser.parse_args()


if __name__ == "__main__":
    main()