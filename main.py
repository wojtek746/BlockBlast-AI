import subprocess
import time
import os
import cv2
import numpy as np
import pytesseract
import requests
import io

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

proc = None

def tap(x, y):
    os.system(f"adb shell input tap {str(x)} {str(y)}")

def swipe(x1, y1, x2, y2, duration_ms=300):
    os.system(f"adb shell input swipe {str(x1)} {str(y1)} {str(x2)} {str(y2)} {str(duration_ms)}")

def sending(filename='screen.png', url='https://logical-raccoon-keen.ngrok-free.app/zdj'):
    screen = cv2.imread(filename)
    send(screen, 94, 644, url)
    send(screen, 213, 644, url)
    send(screen, 333, 644, url)
    send(screen, 452, 644, url)
    send(screen, 571, 644, url)
    send(screen, 690, 644, url)
    send(screen, 809, 644, url)
    send(screen, 928, 644, url)
    send(screen, 94, 763, url)
    send(screen, 213, 763, url)
    send(screen, 333, 763, url)
    send(screen, 452, 763, url)
    send(screen, 571, 763, url)
    send(screen, 690, 763, url)
    send(screen, 809, 763, url)
    send(screen, 928, 763, url)
    send(screen, 94, 882, url)
    send(screen, 213, 882, url)
    send(screen, 333, 882, url)
    send(screen, 452, 882, url)
    send(screen, 571, 882, url)
    send(screen, 690, 882, url)
    send(screen, 809, 882, url)
    send(screen, 928, 882, url)
    send(screen, 94, 1002, url)
    send(screen, 213, 1002, url)
    send(screen, 333, 1002, url)
    send(screen, 452, 1002, url)
    send(screen, 571, 1002, url)
    send(screen, 690, 1002, url)
    send(screen, 809, 1002, url)
    send(screen, 928, 1002, url)
    send(screen, 94, 1121, url)
    send(screen, 213, 1121, url)
    send(screen, 333, 1121, url)
    send(screen, 452, 1121, url)
    send(screen, 571, 1121, url)
    send(screen, 690, 1121, url)
    send(screen, 809, 1121, url)
    send(screen, 928, 1121, url)
    send(screen, 94, 1240, url)
    send(screen, 213, 1240, url)
    send(screen, 333, 1240, url)
    send(screen, 452, 1240, url)
    send(screen, 571, 1240, url)
    send(screen, 690, 1240, url)
    send(screen, 809, 1240, url)
    send(screen, 928, 1240, url)
    send(screen, 94, 1360, url)
    send(screen, 213, 1360, url)
    send(screen, 333, 1360, url)
    send(screen, 452, 1360, url)
    send(screen, 571, 1360, url)
    send(screen, 690, 1360, url)
    send(screen, 809, 1360, url)
    send(screen, 928, 1360, url)
    send(screen, 94, 1479, url)
    send(screen, 213, 1479, url)
    send(screen, 333, 1479, url)
    send(screen, 452, 1479, url)
    send(screen, 571, 1479, url)
    send(screen, 690, 1479, url)
    send(screen, 809, 1479, url)
    send(screen, 928, 1479, url)

def send(screen, x, y, url):
    fragment = screen[y:y+56, x:x+57]
    cv2.imshow("fragment", fragment)
    key = chr(cv2.waitKey(0))
    cv2.destroyWindow("fragment")
    if key in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        _, buffer = cv2.imencode('.png', fragment)
        image_file = io.BytesIO(buffer.tobytes())
        files = {'file': ('fragment.png', image_file, 'image/png')}
        data = {'description': key}
        response = requests.post(url, files=files, data=data)
        if response.status_code != 200:
            print(response.text)
            send(screen, x, y, url)
    else:
        print('zły numer')
        send(screen, x, y, url)

def get_screenshot(filename='screen.png'):
    os.system("adb shell screencap -p /sdcard/screen.png")
    os.system(f"adb pull /sdcard/screen.png {filename}")
    os.system("adb shell rm /sdcard/screen.png")

def board_to_bool_array(filename='screen.png'):
    start_x = 120
    start_y = 670
    board_img = cv2.imread(filename)
    blank = [66, 36, 25]
    result = []
    for i in range(8):
        row = []
        for j in range(8):
            x = start_x + j * 120
            y = start_y + i * 120
            rgb = board_img[y, x]
            row.append(not np.array_equal(rgb, blank))
        result.append(row)
    return result

def main():
    # get_screenshot()
    # bool_array = board_to_bool_array()
    # for row in bool_array:
    #     print(row)
    #swipe(500, 200, 500, 300, 1)
    sending()

if __name__ == "__main__":
    main()