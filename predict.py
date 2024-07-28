from tensorflow.keras.models import load_model
from scipy.special import softmax
import cv2
from PIL import Image 
import numpy as np
from PIL import ImageDraw
import os
import tensorflow as tf
import warnings


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel('ERROR')

def pre(img):
    # print("shape:",img.shape)
    img = cv2.bitwise_not(img)
    # cv2.imshow("sdf",img)
    # cv2.waitKey(0)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img1 = cv2.resize(img,(32,32))
    # img2 = cv2.resize(img,(300,300))

    # cv2.imshow("img1",img1)
    # # cv2.imshow('img2',img2)
    # cv2.waitKey(0)
    # img = cv2.bitwise_not(img)
    # img23 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    # img2 = Image.fromarray(img23) 
    # # img2.show() 
    # a = '\u092C'
    
    # img = cv2.putText(img,a,(50,50),2,2,(23,124,152))
    # img21 = ImageDraw.Draw(img2) 
    # img21.text((123,123),str(a),(123,132,123))
    # img2.show()

    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    # img1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)[1]
    # cv2.imshow('img2',img1)
    # cv2.waitKey(0)

    if len(img1.shape) == 3:  # Check if it has 3 channels (RGB)
        img1 = np.expand_dims(img1, axis=0)
        
    # print("sdf",img1.shape)
    # print("before")
    warnings.filterwarnings("ignore")
    model = load_model('effModel2.h5')
    # print("after")
    # model2 = load_model('efficientnet_gujarati.h5')
    a = model(img1)
    labels = np.load('labels.npy',allow_pickle=True)
    labels = labels[0]
    # print(np.argmax(a[0]))
#
    l = {0: 'Tho', 1: 'Raa', 2: 'Phe', 3: 'Gam', 4: 'Che', 5: 'Tu', 6: 'He', 7: 'Gnam', 8: 'Dhoo', 9: 'Tham', 10: 'Sam', 11: 'Ve', 12: 'No', 13: 'Mu', 14: 'Bu', 15: 'DDO', 16: 'Dau', 17: 'Zoo', 18: 'Khee', 19: 'TTho', 20: 'Bhi', 21: 'Zai', 22: 'Ji', 23: 'Jo', 24: 'I', 25: 'Shoo', 26: 'TToo', 27: 'Soo', 28: 'Ane', 29: 'Ale', 30: 'Phee', 31: '6_six', 32: 'Tam', 33: 'Kho', 34: 'Naa', 35: 'Bho', 36: 'Sho', 37: 'Pam', 38: 'Tai', 39: 'Na', 40: 'Bo', 41: 'Yi', 42: 'Se', 43: '9_nine', 44: 'Chhoo', 45: 'So', 46: 'Chhai', 47: 'Thau', 48: 'TTai', 49: 'Dhee', 50: 'Pai', 51: 'Ko', 52: 'DDhoo', 53: 'Kshi', 54: 'Dhu', 55: 'Gne', 56: 'Taa', 57: 'Ge', 58: '5_five', 59: 'Kham', 60: 'E', 61: 'Rai', 62: 'Yo', 63: 'Yee', 64: 'DDau', 65: 'Ra', 66: 'Mam', 67: 'Jaa', 68: 'Ta', 69: 'TThau', 70: 'Kam', 71: 'Pe', 72: 'Ghee', 73: 'TThai', 74: 'Gee', 75: 'Hai', 76: 'Da', 77: 'Phau', 78: 'Thi', 79: '4_four', 80: 'Lo', 81: 'Yam', 82: 'Ze', 83: 'SShoo', 84: 'Thoo', 85: 'SShe', 86: 'Dai', 87: 'TThoo', 88: 'Khe', 89: 'Lam', 90: 'Thee', 91: 'DDai', 92: 'Chha', 93: 'Va', 94: 'U', 95: 'Thaa', 96: 'DDhaa', 97: 'Ala', 98: 'Gnai', 99: 'DDee', 100: 'TTaa', 101: 'SShee', 102: 'Vai', 103: 'TThaa', 104: 'Sau', 105: 'Ro', 106: 'Jam', 107: 'Gnee', 108: 'Ghe', 109: 'Mo', 110: 'Chhau', 111: 'Du', 112: 'Anoo', 113: 'Kee', 114: 'Chhe', 115: 'Phoo', 116: 'Alaa', 117: 'Chhu', 118: 'Maa', 119: 'Ano', 120: 'Ghai', 121: 'Tau', 122: 'Be', 123: 'Moo', 124: 'Gnau', 125: 'Daa', 126: 'Loo', 127: 'Mau', 128: 'Mai', 129: 'Kshee', 130: 'TTee', 131: 'DDu', 132: 'Bee', 133: 'Yaa', 134: 'Nam', 135: 'Hu', 136: 'Pee', 137: 'Ne', 138: 'Dho', 139: 'Alau', 140: '1_one', 141: 'Kshau', 142: 'Zaa', 143: 'Su', 144: 'DDhee', 145: 'DDho', 146: 'Khau', 147: 'Ksha', 148: 'Shee', 149: 'Yau', 150: 'Pham', 151: 'Alai', 152: 'Vu', 153: 'DDi', 154: 'DDoo', 155: 'Khoo', 156: 'Ju', 157: 'Dhaa', 158: 'Dha', 159: 'Phai', 160: 'Gu', 161: 'Gnu', 162: 'Gaa', 163: 'Ghaa', 164: 'Nu', 165: '7_seven', 166: 'Koo', 167: 'Dee', 168: 'Khai', 169: 'Baa', 170: 'Anau', 171: 'Di', 172: 'Kha', 173: 'Gi', 174: 'Vam', 175: 'Dhau', 176: 'Doo', 177: 'Phaa', 178: 'To', 179: 'Chau', 180: 'Zu', 181: 'See', 182: 'Shai', 183: 'Khu', 184: 'Gni', 185: 'Ali', 186: 'Ri', 187: 'Phi', 188: 'Si', 189: 'Chi', 190: 'Yai', 191: 'Zi', 192: 'Za', 193: 'Ja', 194: 'Pha', 195: 'Bi', 196: 'Haa', 197: 'Noo', 198: 'Too', 199: '8_eight', 200: 'Gho', 201: 'Ke', 202: 'Hee', 203: 'DDam', 204: 'TTha', 205: 'Anai', 206: 'Phu', 207: 'TThu', 208: 'Te', 209: 'SShau', 210: 'Chhee', 211: 'Goo', 212: 'Shaa', 213: 'Go', 214: 'Ya', 215: 'Pu', 216: 'Bha', 217: 'Saa', 218: 'Chai', 219: 'SShi', 220: 'Anam', 221: 'She', 222: 'TTam', 223: 'Anu', 224: 'Ba', 225: 'TTu', 226: 'Dhai', 227: 'Vee', 228: 'Rau', 229: 'TTo', 230: 'Gna', 231: 'A', 232: 'Shau', 233: 'Me', 234: 'Ma', 235: 'TThi', 236: 'DDhu', 237: 'Tha', 238: 'TTi', 239: 'Bam', 240: 'O', 241: 'Mi', 242: 'Ghoo', 243: 'Re', 244: 'Lee', 245: 'Mee', 246: 'Chham', 247: 'Alu', 248: 'Ani', 249: 'Bai', 250: 'Sham', 251: 'DDham', 252: 'Cho', 253: 'Po', 254: 'Ti', 255: 'Pau', 256: 'Bhu', 257: 'SSha', 258: 'Ai', 259: 'Voo', 260: 'Zee', 261: '3_three', 262: 'Jau', 263: 'Je', 264: 'Pa', 265: 'Jee', 266: 'Kaa', 267: 'Nee', 268: 'Ho', 269: 'DDhe', 270: 'Choo', 271: 'TTau', 272: 'Hoo', 273: 'Roo', 274: 'Li', 275: 'Ru', 276: 'Ksh', 277: 'Sha', 278: 'Yu', 279: 'Ksham', 280: 'Ghi', 281: 'Dam', 282: 'Nai', 283: 'Yoo', 284: 'Hau', 285: 'Do', 286: 'SShai', 287: 'Ga', 288: 'Anee', 289: 'TTa', 290: 'Gha', 291: 'Lau', 292: 'Vau', 293: 'Jai', 294: 'Au', 295: 'Boo', 296: 'Vo', 297: 'TThe', 298: 'TTe', 299: 'SShu', 300: 'SSham', 301: 'Ghau', 302: 'Vaa', 303: 'Lai', 304: 'Anaa', 305: 'DDaa', 306: 'Alo', 307: 'Tee', 308: 'Am', 309: 'Ham', 310: 'Ksho', 311: '0_zero', 312: 'TTham', 313: 'Alee', 314: 'Lu', 315: 'Pi', 316: 'Kshai', 317: 'Poo', 318: 'Zau', 319: 'Shi', 320: 'Dham', 321: 'Cham', 322: 'Chee', 323: 'Gnaa', 324: 'AA', 325: 'Ku', 326: 'Bhoo', 327: 'Ni', 328: 'Chaa', 329: 'Shu', 330: 'Kshu', 331: 'OO', 332: 'Cha', 333: 'Bhe', 334: 'Zam', 335: 'Ana', 336: 'Vi', 337: 'Bhaa', 338: 'Le', 339: 'Bau', 340: 'De', 341: 'DDe', 342: 'SSh', 343: 'Aloo', 344: 'Bhau', 345: 'The', 346: 'Dhe', 347: 'Thu', 348: 'TThee', 349: 'Bhee', 350: 'Chhaa', 351: 'Gai', 352: 'Kshoo', 353: 'Sai', 354: 'Ki', 355: 'Ree', 356: 'Gau', 357: 'DDa', 358: 'Kau', 359: 'Alam', 360: 'Gno', 361: 'DDhi', 362: 'Thai', 363: 'Khaa', 364: 'Bham', 365: 'Kai', 366: 'Dhi', 367: 'Kshe', 368: 'Gham', 369: 'Hi', 370: 'Pho', 371: 'Ye', 372: 'Chho', 373: 'EE', 374: 'Nau', 375: 'Khi', 376: 'Ram', 377: '2_two', 378: 'Bhai', 379: 'Gnoo', 380: 'Ghu', 381: 'Chu', 382: 'Ka', 383: 'Paa', 384: 'Chhi', 385: 'DDha', 386: 'SSho', 387: 'La', 388: 'Ha', 389: 'Zo', 390: 'Joo', 391: 'Laa', 392: 'DDhau', 393: 'DDhai', 394: 'Sa'}
    # print(l[np.argmax(a[0])])
    return l[np.argmax(a[0])]
    # print(labels,type(labels))

    # print(a[0],b,labels,labels[np.argmax(a[0])],labels[np.argmax(b[0])],img)
    # print(labels())