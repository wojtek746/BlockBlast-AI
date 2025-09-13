import time
import torch
import os
import cv2
import numpy as np
from PIL import Image

proc = None

fragments = []
coords = []
sent = {}

device, numberAI = None, None
def init_number_ai():
    import pytesseract
    from learnNumberAI import NumberNet

    global device, numberAI

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    device = torch.device("cpu")
    numberAI = NumberNet().to(device)

    numberAI.load_state_dict(torch.load("number_net.pth", map_location=device))
    numberAI.eval()

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

def number(fragment):
    global numberAI

    img = cv2.cvtColor(fragment, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    tensor = numberAI.transform(pil_img).unsqueeze(0).to(device)

    numberAI.eval()
    with torch.no_grad():
        outputs = numberAI(tensor)
        _, predicted = torch.max(outputs, 1)

    return int(predicted.item())

blue = [(148, 81, 58), (148, 85, 58), (148, 85, 66), (140, 81, 58), (140, 73, 49), (132, 69, 49), (140, 77, 58), (132, 65, 41), (132, 65, 49), (140, 77, 49), (156, 85, 66)]

def isin(board, x, y):
    global blue
    return tuple(board[y, x]) in blue

def shape(b, n):
    global blue
    match(n):
        case 1:
            x = 198
        case 2:
            x = 513
        case 3:
            x = 828
        case _:
            return None
    y = 1822

    if isin(b, x+5, y+5) and isin(b, x+49, y+5) and isin(b, x+5, y+49) and isin(b, x+49, y+49):
        return None

    if isin(b, x-5, y+5) and isin(b, x-5, y+49) and isin(b, x+59, y+5) and isin(b, x+59, y+49): #1xX
        if isin(b, x+5, y-5) and isin(b, x+49, y-5) and isin(b, x+5, y+59) and isin(b, x+49, y+59):
            return [[True, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-32) and isin(b, x+49, y-32) and isin(b, x+5, y+86) and isin(b, x+49, y+86):
            return [[True, False, False, False, False], [True, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-59) and isin(b, x+49, y-59) and isin(b, x+5, y+113) and isin(b, x+49, y+113):
            return [[True, False, False, False, False], [True, False, False, False, False], [True, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-86) and isin(b, x+49, y-86) and isin(b, x+5, y+140) and isin(b, x+49, y+140):
            return [[True, False, False, False, False], [True, False, False, False, False], [True, False, False, False, False], [True, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-113) and isin(b, x+49, y-113) and isin(b, x+5, y+167) and isin(b, x+49, y+167):
            return [[True, False, False, False, False], [True, False, False, False, False], [True, False, False, False, False], [True, False, False, False, False], [True, False, False, False, False]]
    if isin(b, x-32, y+5) and isin(b, x-32, y+49) and isin(b, x+86, y+5) and isin(b, x+86, y+49): #2xX
        if isin(b, x+5, y-5) and isin(b, x+49, y-5) and isin(b, x+5, y+59) and isin(b, x+49, y+59):
            return [[True, True, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-32) and isin(b, x+49, y-32) and isin(b, x+5, y+86) and isin(b, x+49, y+86):
            return [[not isin(b, x, y), not isin(b, x+54, y), False, False, False], [not isin(b, x, y+54), not isin(b, x+54, y+54), False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-59) and isin(b, x+49, y-59) and isin(b, x+5, y+113) and isin(b, x+49, y+113):
            return [[not isin(b, x, y-27), not isin(b, x+54, y-27), False, False, False], [not isin(b, x, y+27), not isin(b, x+54, y+27), False, False, False], [not isin(b, x, y+81), not isin(b, x+54, y+81), False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-86) and isin(b, x+49, y-86) and isin(b, x+5, y+140) and isin(b, x+49, y+140):
            return [[not isin(b, x, y-54), not isin(b, x+54, y-54), False, False, False], [not isin(b, x, y), not isin(b, x+54, y), False, False, False], [not isin(b, x, y+54), not(b, x+54, y+54), False, False, False], [not isin(b, x, y+108), not isin(b, x+54, y+108), False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-113) and isin(b, x+49, y-113) and isin(b, x+5, y+167) and isin(b, x+49, y+167):
            return [[not isin(b, x, y-81), not isin(b, x+54, y-81), False, False, False], [not isin(b, x, y-27), not isin(b, x+54, y-27), False, False, False], [not isin(b, x, y+27), not(b, x+54, y+27), False, False, False], [not isin(b, x, y+81), not isin(b, x+54, y+81), False, False, False], [not isin(b, x, y+135), not isin(b, x+54, y+135), False, False, False]]
    if isin(b, x-59, y+5) and isin(b, x-59, y+49) and isin(b, x+113, y+5) and isin(b, x+113, y+49): #3xX
        if isin(b, x+5, y-5) and isin(b, x+49, y-5) and isin(b, x+5, y+59) and isin(b, x+49, y+59):
            return [[True, True, True, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-32) and isin(b, x+49, y-32) and isin(b, x+5, y+86) and isin(b, x+49, y+86):
            return [[not isin(b, x-27, y), not isin(b, x+27, y), not isin(b, x+81, y), False, False], [not isin(b, x-27, y+54), not isin(b, x+27, y+54), not isin(b, x+81, y+54), False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-59) and isin(b, x+49, y-59) and isin(b, x+5, y+113) and isin(b, x+49, y+113):
            return [[not isin(b, x-27, y-27), not isin(b, x+27, y-27), not isin(b, x+81, y-27), False, False], [not isin(b, x-27, y+27), not isin(b, x+27, y+27), not isin(b, x+81, y+27), False, False], [not isin(b, x-27, y+81), not isin(b, x+27, y+81), not isin(b, x+81, y+81), False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-86) and isin(b, x+49, y-86) and isin(b, x+5, y+140) and isin(b, x+49, y+140):
            return 3, 4
        if isin(b, x+5, y-113) and isin(b, x+49, y-113) and isin(b, x+5, y+167) and isin(b, x+49, y+167):
            return 3, 5
    if isin(b, x-86, y+5) and isin(b, x-86, y+49) and isin(b, x+140, y+5) and isin(b, x+140, y+49): #4xX
        if isin(b, x+5, y-5) and isin(b, x+49, y-5) and isin(b, x+5, y+59) and isin(b, x+49, y+59):
            return 4, 1
        if isin(b, x+5, y-32) and isin(b, x+49, y-32) and isin(b, x+5, y+86) and isin(b, x+49, y+86):
            return 4, 2
        if isin(b, x+5, y-59) and isin(b, x+49, y-59) and isin(b, x+5, y+113) and isin(b, x+49, y+113):
            return 4, 3
        if isin(b, x+5, y-86) and isin(b, x+49, y-86) and isin(b, x+5, y+140) and isin(b, x+49, y+140):
            return 4, 4
        if isin(b, x+5, y-113) and isin(b, x+49, y-113) and isin(b, x+5, y+167) and isin(b, x+49, y+167):
            return 4, 5
    if isin(b, x-113, y+5) and isin(b, x-113, y+49) and isin(b, x+167, y+5) and isin(b, x+167, y+49): #5xX
        if isin(b, x+5, y-5) and isin(b, x+49, y-5) and isin(b, x+5, y+59) and isin(b, x+49, y+59):
            return 5, 1
        if isin(b, x+5, y-32) and isin(b, x+49, y-32) and isin(b, x+5, y+86) and isin(b, x+49, y+86):
            return 5, 2
        if isin(b, x+5, y-59) and isin(b, x+49, y-59) and isin(b, x+5, y+113) and isin(b, x+49, y+113):
            return 5, 3
        if isin(b, x+5, y-86) and isin(b, x+49, y-86) and isin(b, x+5, y+140) and isin(b, x+49, y+140):
            return 5, 4
        if isin(b, x+5, y-113) and isin(b, x+49, y-113) and isin(b, x+5, y+167) and isin(b, x+49, y+167):
            return 5, 5


def board_to_number_array(filename='screen.png'):
    xs = [94,213,333,452,571,690,809,928]
    ys = [644,763,882,1002,1121,1240,1360,1479]
    board_img = cv2.imread(filename)
    result = []
    for y in ys:
        row = []
        for x in xs:
            fragment = board_img[y:y + 56, x:x + 57]
            n = number(fragment)
            if n == 0:
                row.append(0)
            else:
                row.append(1/n)
        result.append(row)
    return result

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
    get_screenshot()
    # bool_array = board_to_bool_array()
    # for row in bool_array:
    #     print(row)
    # init_number_ai()
    # number_array = board_to_number_array()
    # for row in number_array:
    #     print(row)
    # board = cv2.imread('screen.png')
    # for i in range(1, 4):
    #     array = shape(board, i)
    #     if array is not None:
    #         for row in array:
    #             print(row)
    #     print("\n\n\n")
    #swipe(500, 200, 500, 300, 1)
    sending()

if __name__ == "__main__":
    main()