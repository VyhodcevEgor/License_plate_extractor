import matplotlib.pyplot as plt
import pytesseract
import cv2
from config import cmd_path, haar_classifier_path, pytesseract_string_config


pytesseract.pytesseract.tesseract_cmd = cmd_path


def open_image(img_path):
    license_plate = cv2.imread(img_path)
    license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB)

    return license_plate


def enlarge_img(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    plt.axis('off')
    resized_img = cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)

    return resized_img


def extract_license_plate(img, haar_cascade):
    license_plate_rects = haar_cascade.detectMultiScale(
        img, scaleFactor=1.1, minNeighbors=5
    )
    for x, y, width, height in license_plate_rects:
        license_plate_img = img[y + 15:y + height - 10, x + 15:x + width - 11]

    return license_plate_img


def main():
    license_plate_rgb = open_image(img_path='5.jpg')
    license_plate_haar_cascade = cv2.CascadeClassifier(haar_classifier_path)
    extracted_license_plate = extract_license_plate(
        license_plate_rgb, license_plate_haar_cascade
    )
    extracted_license_plate = enlarge_img(extracted_license_plate, 150)

    extracted_license_plate_gray = cv2.cvtColor(extracted_license_plate,
                                                cv2.COLOR_RGB2GRAY)
    plt.axis('off')
    plt.imshow(extracted_license_plate_gray, cmap='gray')
    plt.show()

    string_plate = pytesseract.image_to_string(
        extracted_license_plate_gray,
        config=pytesseract_string_config
    )
    if string_plate == '':
        print(f"Номер автомобиля не распознан!")
    else:
        print(f"Номер автомобиля: {string_plate}")


if __name__ == '__main__':
    main()
