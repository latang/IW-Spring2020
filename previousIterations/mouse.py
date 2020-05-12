import pyautogui
import time
# import keyboard

def main():
    # print(pyautogui.size())
    pyautogui.moveTo(350, 230, duration=1)
    pyautogui.click()
    try:
        while True:
            pyautogui.moveTo(380, 180, duration=1)
            pyautogui.click()
            pyautogui.moveTo(455, 230, duration=1)
            pyautogui.click()
            pyautogui.moveTo(650, 230, duration=1)
            pyautogui.moveTo(661, 254, duration=1)
            pyautogui.click()

            # if keyboard.is_pressed('q'):  # if key 'q' is pressed
            #     print('You Pressed A Key!')
            # break
    except KeyboardInterrupt:
        print('\n')
    # pyautogui.click()
    # drag = 18
    # for i in range(30):
    #     pyautogui.moveRel(drag, 0)
    #
    # while True:
    #     start = time.time()
    #     end = time.time()
    #     elapsed = end - start
    #     if elapsed < 10000:
    #         x, y = pyautogui.position()
    #         positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
    #         print(positionStr)



if __name__ == '__main__':
    main()
