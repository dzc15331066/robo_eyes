#!/usr/bin/env python
import face_recognition
import cv2
import sys


def facereco(path):
    unknown_image = face_recognition.load_image_file(path)
    #print(unknown_image)
    unknown_encoding = face_recognition.face_encodings(unknown_image)
    print(unknown_encoding)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "invalid number of args"
        sys.exit(1)
    facereco(sys.argv[1])