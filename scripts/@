import face_recognition
import cv2


def facereco(img_path):

    known_image = face_recognition.load_image_file("shenghui.jpg")
    unknown_image = face_recognition.load_image_file(img_path)

    biden_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
    if results == [True]:
        return 'shenghui'
    else:
        known_image = face_recognition.load_image_file("andy.jpg")
        andy_encoding = face_recognition.face_encodings(known_image)[0]
        results = face_recognition.compare_faces([andy_encoding], unknown_encoding)
        if results == [True]:
            print("andy")
            return 'andy' 
        else:
            return 'unKnown'
if __name__ == '__main__':
    #res = facereco('upload_img.jpg')
    res = facereco('upload_img.jpg')
    print(res)
