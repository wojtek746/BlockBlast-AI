import subprocess
import time
import os
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

proc = None

fragments = []
coords = []
sent = {}

def tap(x, y):
    os.system(f"adb shell input tap {str(x)} {str(y)}")

def swipe(x1, y1, x2, y2, duration_ms=300):
    os.system(f"adb shell input swipe {str(x1)} {str(y1)} {str(x2)} {str(y2)} {str(duration_ms)}")

def sending(filename='screen.png'):
    screen = cv2.imread(filename)

    xs = [94,213,333,452,571,690,809,928]
    ys = [644,763,882,1002,1121,1240,1360,1479]

    global fragments, coords, sent
    fragments = []
    coords = []
    sent = {}

    for yi, y in enumerate(ys):
        row_frag = []
        row_coords = []
        for xi, x in enumerate(xs):
            frag = screen[y:y + 56, x:x + 57]
            row_frag.append(frag)
            row_coords.append((x, y))
        fragments.append(row_frag)
        coords.append(row_coords)

    grid = np.zeros((len(ys) * 60, len(xs) * 60, 3), dtype=np.uint8)
    for i in range(len(ys)):
        for j in range(len(xs)):
            grid[i * 60:i * 60 + 56, j * 60:j * 60 + 57] = fragments[i][j]

    cv2.imshow("board", grid)
    cv2.setMouseCallback("board", on_click)

    while True:
        if cv2.waitKey(50) == 13:  # Enter
            break

    cv2.destroyAllWindows()

    print("wysyłanie reszty z 0 (to może trochę potrwać)")

    for i in range(len(ys)):
        for j in range(len(xs)):
            if (coords[i][j] not in sent):
                send(fragments[i][j], '0', *coords[i][j])

    print("wysłano wszystko")

def send(fragment, key, x, y):
    import response
    gray = cv2.cvtColor(fragment, cv2.COLOR_BGR2GRAY)
    response = response.response(gray, key)
    if response.status_code != 200:
        print(response.text)
        return False
    sent[(x, y)] = True
    if key != '0':
        print(f"wysłano {key}")
    return True

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = (y // 60, x // 60)
        if idx[0] < len(fragments) and idx[1] < len(fragments[0]):
            fragment = fragments[idx[0]][idx[1]]
            x, y = coords[idx[0]][idx[1]]
            print(f"Klik {idx}, czekam na cyfrę 1-9")
            key = chr(cv2.waitKey(0))
            if key in '123456789':
                send(fragment, key, x, y)

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