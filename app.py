from PIL import ImageGrab
from time import sleep
import numpy as np
import pydirectinput
import cv2
from model.efficientNet import EfficientNet
import torch
from utils import ProcessImage

# Params
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSIES = 8
IMG_SIZE = 240
PATH = "./model/weighted.pt"

def main():
    # Loading model
    print("Loading model....")
    model = EfficientNet(NUM_CLASSIES)
    model.load_state_dict(torch.load(PATH,map_location = DEVICE))
    model = model.to(DEVICE)
    loaded_model = ProcessImage(IMG_SIZE, model, DEVICE)
    print("Loading done!\n")

    print("-"*10,"CHOI GTA SA DUM NGUOI CUT TAY TRONG","-"*10)
    sleep(1)
    for i in range(3, 0 ,-1):
        print(i)
        sleep(1)

    action_history,action = [],[]

    try:
        while True:
            screen = ImageGrab.grab(bbox=(0,0,800,600))

            #cv2.imshow("window", cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2RGB))
            action = loaded_model.predict(screen)

            for key in action_history:
                pydirectinput.keyUp(key)
            for key in action:
                pydirectinput.keyDown(key)
            action_history = action
            
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()