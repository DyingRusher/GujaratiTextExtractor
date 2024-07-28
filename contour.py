import cv2
import numpy as np
import predict

l = {'0_zero': '૦', '1_one': '૧', '2_two': '૨', '3_three': '૩', '4_four': '૪', '5_five': '૫', '6_six': '૬', '7_seven': '૭', '8_eight': '૮', '9_nine': '૯', 'A': 'અ', 'AA': 'આ', 'Ai': 'ઐ', 'Ala': 'ળ', 'Alaa': 'ળા', 'Alai': 'ળૈ', 'Alam': 'ળં', 'Alau': 'ળૌ', 'Ale': 'ળે', 'Alee': 'ળી', 'Ali': 'ળિ', 'Alo': 'ળો', 'Aloo': 'ળૂ', 'Alu': 'ળુ', 'Am': 'અં', 'Ana': 'ણ', 'Anaa': 'ણા', 'Anai': 'ણૈ', 'Anam': 'ણં', 'Anau': 'ણૌ', 'Ane': 'ણે', 'Anee': 'ણી', 'Ani': 'ણિ', 'Ano': 'ણૉ', 'Anoo': 'ણૂ', 'Anu': 'ણુ', 'Au': 'ઔ', 'Ba': 'બ', 'Baa': 'બા', 'Bai': 'બૈ', 'Bam': 'બં', 'Bau': 'બૌ', 'Be': 'બે', 'Bee': 'બી', 'Bha': 'ભ', 'Bhaa': 'ભા', 'Bhai': 'ભૈ', 'Bham': 'ભં', 'Bhau': 'ભૌ', 'Bhe': 'ભે', 'Bhee': 'ભી', 'Bhi': 'ભિ', 'Bho': 'ભો', 'Bhoo': 'ભૂ', 'Bhu': 'ભુ', 'Bi': 'બિ', 'Bo': 'બો', 'Boo': 'બૂ', 'Bu': 'બુ', 'Cha': 'ચ', 'Chaa': 'ચા', 'Chai': 'ચૈ', 'Cham': 'ચં', 'Chau': 'ચૌ', 'Che': 'ચે', 'Chee': 'ચી', 'Chha': 'છ', 'Chhaa': 'છા', 'Chhai': 'છૈ', 'Chham': 'છં', 'Chhau': 'છૌ', 'Chhe': 'છે', 'Chhee': 'છી', 'Chhi': 'છિ', 'Chho': 'છો', 'Chhoo': 'છૂ', 'Chhu': 'છુ', 'Chi': 'ચિ', 'Cho': 'ચો', 'Choo': 'ચૂ', 'Chu': 'ચુ', 'DDO': 'ડો', 'DDa': 'ડ', 'DDaa': 'ડા', 'DDai': 'ડૈ', 'DDam': 'ડં', 'DDau': 'ડૌ', 'DDe': 'ડે', 'DDee': 'ડી', 'DDha': 'ઢ', 'DDhaa': 'ઢા', 'DDhai': 'ઢૈ', 'DDham': 'ઢં', 'DDhau': 'ઢૌ', 'DDhe': 'ઢે', 'DDhee': 'ઢી', 'DDhi': 'ઢિ', 'DDho': 'ઢો','DDhoo': 'ઢૂ', 'DDhu': 'ઢુ', 'DDi': 'ડિ', 'DDoo': 'ડૂ', 'DDu': 'ડુ', 'Da': 'દ', 'Daa': 'દા', 'Dai': 'દૈ', 'Dam': 'દં', 'Dau': 'દૌ', 'De': 'દે', 'Dee': 'દી', 'Dha': 'ઘ','Dhaa': 'ઘા',   'Dhai': 'ઘૈ',  'Dham': 'ઘં',   'Dhau': 'ઘૌ',   'Dhe': 'ઘે',    'Dhee': 'ઘી','Dhi': 'ઘિ',   'Dho': 'ઘો',   'Dhoo': 'ઘૂ',  'Dhu': 'ઘુ',    'Di': 'દિ',     'Do': 'દો',    'Doo': 'દૂ',   'Du': 'દુ',     'E': 'એ',      'EE': 'ઈ',     'Ga': 'ગ','Gaa': 'ગા',  'Gai': 'ગૈ',    'Gam': 'ગં',  'Gau': 'ગૌ',  'Ge': 'ગે',     'Gee': 'ગી', 'Gha': 'ઘ', 'Ghaa': 'ઘા', 'Ghai': 'ઘૈ', 'Gham': 'ઘં', 'Ghau': 'ઘૌ', 'Ghe': 'ઘે', 'Ghee': 'ઘી', 'Ghi': 'ઘિ', 'Gho': 'ઘો', 'Ghoo': 'ઘૂ', 'Ghu': 'ઘુ', 'Gi': 'ગિ', 'Gna': 'જ્ઞ', 'Gnaa': 'જ્ઞા', 'Gnai': 'જ્ઞૈ', 'Gnam': 'જ્ઞં', 'Gnau': 'જ્ઞૌ', 'Gne': 'જ્ઞે', 'Gnee': 'જ્ઞી', 'Gni': 'જ્ઞિ', 'Gno': 'જ્ઞો', 'Gnoo': 'જ્ઞૂ', 'Gnu': 'જ્ઞુ', 'Go': 'ગો', 'Goo': 'ગૂ', 'Gu': 'ગુ', 'Ha': 'હ', 'Haa': 'હા', 'Hai': 'હૈ', 'Ham': 'હં', 'Hau': 'હૌ', 'He': 'હે', 'Hee': 'હી', 'Hi': 'હિ', 'Ho': 'હો', 'Hoo': 'હૂ', 'Hu': 'હુ', 'I': 'ઇ', 'Ja': 'જ', 'Jaa': 'જા', 'Jai': 'જૈ', 'Jam': 'જં', 'Jau': 'જૌ', 'Je': 'જે', 'Jee': 'જી', 'Ji': 'જિ', 'Jo': 'જો', 'Joo': 'જૂ', 'Ju': 'જુ', 'Ka': 'ક', 'Kaa': 'કા', 'Kai': 'કૈ', 'Kam': 'કં', 'Kau': 'કૌ', 'Ke': 'કે', 'Kee': 'કી', 'Kha': 'ખ', 'Khaa': 'ખા', 'Khai': 'ખૈ', 'Kham': 'ખં', 'Khau': 'ખૌ', 'Khe': 'ખે', 'Khee': 'ખી', 'Khi': 'ખિ', 'Kho': 'ખો', 'Khoo': 'ખૂ', 'Khu': 'ખુ' ,'Ki': 'કિ', 'Ko': 'કો', 'Koo': 'કૂ', 'Ksh': 'ક્ષ', 'Ksha': 'ક્ષા', 'Kshai': 'ક્ષૈ', 'Ksham': 'ક્ષં', 'Kshau': 'ક્ષૌ', 'Kshe': 'ક્ષે', 'Kshee': 'ક્ષી', 'Kshi': 'ક્ષિ', 'Ksho': 'ક્ષો', 'Kshoo': 'ક્ષૂ', 'Kshu': 'ક્ષુ', 'Ku': 'કુ', 'La': 'લ', 'Laa': 'લા', 'Lai': 'લૈ', 'Lam': 'લં', 'Lau': 'લૌ', 'Le':'લે', 'Lee': 'લી', 'Li': 'લિ', 'Lo': 'લો', 'Loo': 'લૂ', 'Lu': 'લુ', 'Ma': 'મ', 'Maa': 'મા', 'Mai': 'મૈ', 'Mam': 'મં', 'Mau': 'મૌ', 'Me': 'મે', 'Mee': 'મી', 'Mi': 'મિ', 'Mo': 'મો', 'Moo': 'મૂ', 'Mu': 'મુ', 'Na': 'ન', 'Naa': 'ના', 'Nai': 'નૈ', 'Nam': 'નં', 'Nau': 'નૌ', 'Ne': 'ને', 'Nee': 'ની', 'Ni': 'નિ', 'No': 'નો', 'Noo': 'નૂ', 'Nu': 'નુ', 'O': 'ઓ', 'OO': 'ઊ', 'Pa': 'પ', 'Paa': 'પા', 'Pai': 'પૈ', 'Pam': 'પં', 'Pau': 'પૌ', 'Pe': 'પે', 'Pee': 'પી', 'Pha': 'ફ', 'Phaa': 'ફા', 'Phai': 'ફૈ', 'Pham': 'ફં', 'Phau': 'ફૌ', 'Phe': 'ફે', 'Phee': 'ફી', 'Phi': 'ફિ', 'Pho': 'ફો', 'Phoo': 'ફૂ', 'Phu': 'ફુ', 'Pi': 'પિ', 'Po': 'પો', 'Poo': 'પૂ', 'Pu': 'પુ', 'Ra': 'ર', 'Raa': 'રા', 'Rai': 'રૈ', 'Ram': 'રં', 'Rau': 'રૌ', 'Re': 'રે', 'Ree': 'રી', 'Ri': 'રિ', 'Ro': 'રો', 'Roo': 'રૂ', 'Ru': 'રુ', 'SSh': 'ષ', 'SSha': 'ષા', 'SShai': 'ષૈ', 'SSham': 'ષં ', 'SShau': 'ષૌ', 'SShe': 'ષે', 'SShee': 'ષી', 'SShi': 'ષિ', 'SSho': 'ષો', 'SShoo': 'ષૂ', 'SShu': 'ષુ', 'Sa': 'સ', 'Saa': 'સા', 'Sai': 'સૈ', 'Sam': 'સં ', 'Sau': 'સૌ', 'Se': 'સે', 'See': 'સી', 'Sha': 'શ', 'Shaa': 'શા', 'Shai': 'શૈ', 'Sham': 'શં', 'Shau': 'શૌ', 'She': 'શે', 'Shee': 'શી', 'Shi': 'શિ', 'Sho': 'શો', 'Shoo': 'શૂ', 'Shu': 'શુ', 'Si': 'સિ', 'So': 'સો', 'Soo': 'સૂ', 'Su': 'સુ', 'TTa': 'ટ', 'TTaa': 'ટા', 'TTai': 'ટૈ', 'TTam': 'ટં', 'TTau': 'ટૌ', 'TTe': 'ટે', 'TTee': 'ટી', 'TTha':'ઠ', 'TThaa': 'ઠા', 'TThai': 'ઠૈ', 'TTham': 'ઠં', 'TThau': 'ઠૌ', 'TThe': 'ઠે', 'TThee': 'ઠી', 'TThi': 'ઠિ', 'TTho': 'ઠો', 'TThoo': 'ઠૂ', 'TThu': 'ઠુ', 'TTi': 'ટિ', 'TTo': 'ટો', 'TToo': ' ટૂ', 'TTu': 'ટુ', 'Ta': 'ત', 'Taa': 'તા', 'Tai': 'તૈ', 'Tam': 'તં', 'Tau': 'તૌ', 'Te': 'તે', 'Tee': 'તી', 'Tha': 'થ', 'Thaa': 'થા', 'Thai': 'થૈ', 'Tham': 'થં', 'Thau': 'થૌ', 'The': 'થે', 'Thee': 'થી', 'Thi': 'થિ', 'Tho': 'થો', 'Thoo': 'થૂ', 'Thu': 'થુ', 'Ti': 'તિ', 'To': 'તો', 'Too': 'તૂ', 'Tu': 'તુ', 'U': 'ઉ', 'Va': 'વ', 'Vaa': 'વા', 'Vai': 'વૈ', 'Vam': 'વં', 'Vau': 'વૌ', 'Ve': 'વે', 'Vee': 'વી', 'Vi': 'વિ', 'Vo': 'વો', 'Voo': 'વૂ', 'Vu': 'વુ', 'Ya': 'ય', 'Yaa': 'યા', 'Yai': 'યૈ', 'Yam': 'યં', 'Yau': 'યૌ', 'Ye': 'યે', 'Yee': 'યી', 'Yi': 'યિ', 'Yo': 'યો', 'Yoo': ' યૂ', 'Yu': 'યુ', 'Za': 'ઝ', 'Zaa': 'ઝા', 'Zai': 'ઝૈ', 'Zam': 'ઝં', 'Zau': 'ઝૌ', 'Ze': 'ઝે', 'Zee': 'ઝી', 'Zi': 'ઝિ', 'Zo': 'ઝો','Zoo': 'ઝૂ', 'Zu': 'ઝુ'}

def sep_single_cha(img):
    # img = cv2.imread('C:/Users/ASUS/Desktop/gurati_text_extractor/datasets/yolo_dataset/guj_pic.png')
    h,w = img.shape
    # h = 5
    # w = 6
    # img = cv2.bitwise_not(img)
    img2 = cv2.Canny(img,500,1000)
    
    # print("height",h,wimg)
    img2 = cv2.resize(img2,(int(w*3.5),h*5))
    img = cv2.resize(img,(int(w*3.5),h*5))
    # cv2.imshow("normal",img)
    # cv2.imshow("canny",img2)
    # cv2.waitKey(0)

    # img2 = cv2.resize(img2,(450,300))
    # img = cv2.resize(img,(450,300))

    area = h*5*w*3.5
    
    cont,he  = cv2.findContours(img2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    t_cont = len(cont)
    # print("cont",cont)
    # print("number of countours found:" + str(t_cont))

    co = 0
    ig = []
    xmin_a,ymin_a,xmax_a,ymax_a,xdiff,ydiff,xdiff_a,ydiff_a = 0,0,0,0,[],[],0,0
    xmin_ar,xmax_ar,ymin_ar,ymax_ar = [],[],[],[]

    # print(cont)
    for c  in cont: #for calculating xmin_a,ymin_a,xmax_a,ymax_a

        # print(h,w,_,"SFD")
        xmin = int(w*3.5)
        ymin = int(h*5)
        xmax = 0
        ymax = 0

        for c1 in c:
            if xmin > c1[0][0]:
                xmin = c1[0][0]

            if xmax < c1[0][0]:
                xmax = c1[0][0]
            
            if ymin > c1[0][1]:
                ymin = c1[0][1]
            if ymax < c1[0][1]:
                ymax = c1[0][1]

        xmin_a  = xmin_a + xmin   
        ymin_a  = ymin_a + ymin 
        xmax_a  = xmax_a + xmax 
        ymax_a  = ymax_a + ymax 
        xdiff_a = xdiff_a + xmax - xmin
        ydiff_a = ydiff_a + ymax - ymin

        xdiff.append(xmax-xmin)
        ydiff.append(ymax-ymin)
        xmin_ar.append(xmin)
        xmax_ar.append(xmax)
        ymin_ar.append(ymin)
        ymax_ar.append(ymax)

    xmin_a = xmin_a/int(t_cont)
    ymin_a = xmin_a/int(t_cont)
    xmax_a = xmax_a/int(t_cont) 
    ymax_a = ymax_a/int(t_cont)
    xdiff_a  = xdiff_a/int(t_cont)
    ydiff_a = ydiff_a/int(t_cont)
   
    for i in range(len(xmin_ar)):
        #area is small
        if ((xmax_ar[i] - xmin_ar[i])*(ymax_ar[i] - ymin_ar[i])) < area*0.017:
            # print(i,((xmax_ar[i] - xmin_ar[i])*(ymax_ar[i] - ymin_ar[i]))/area)
            ig.append(i)
            
    # print("small area",ig)
   
    all_ar = [[],[],[],[]]

    for i in range(len(xmin_ar)):
        if i not in ig:
            all_ar[0].append(xmin_ar[i])
            all_ar[1].append(ymin_ar[i])
            all_ar[2].append(xmax_ar[i])
            all_ar[3].append(ymax_ar[i])

    #sorting accourding to xmin
    all_ar = np.array(all_ar)
    so = np.argsort(all_ar[0])
    all_ar2 = [[],[],[],[]]
    for i in range(len(all_ar[0])):
        all_ar2[0].append(all_ar[0][so[i]])
        all_ar2[1].append(all_ar[1][so[i]])
        all_ar2[2].append(all_ar[2][so[i]])
        all_ar2[3].append(all_ar[3][so[i]])
    all_ar = all_ar2

    
    # print(all_ar)

    # 2 if area of intersection is big then add both
    for i in range(len(all_ar[0])):
        # print("len",len(all_ar[0]))
        for j in range(len(all_ar[0])): 
            if i is not  j and i < len(all_ar[0]) and j  < len(all_ar[0]):
                # print(i,j)
                intersection_area =  max(0,min(all_ar[2][i],all_ar[2][j]) - max(all_ar[0][i],all_ar[0][j]))*max(0,min(all_ar[3][i],all_ar[3][j]) - max(all_ar[1][i],all_ar[1][j]))
                intersection_area = intersection_area
                area1 = (all_ar[2][i] - all_ar[0][i]) * (all_ar[3][i] - all_ar[1][i])
                area2 = (all_ar[2][j] - all_ar[0][j]) * (all_ar[3][j] - all_ar[1][j])

                if intersection_area/(area1 + area2 - intersection_area) > 0.09:
                    # print("added",i,j)

                    if i > j:
                        xmin = min(all_ar[0][i],all_ar[0][j])
                        ymin = min(all_ar[1][i],all_ar[1][j])
                        xmax = max(all_ar[2][i],all_ar[2][j])
                        ymax = max(all_ar[3][i],all_ar[3][j])

                        all_ar[0][j] = xmin
                        all_ar[1][j] = ymin
                        all_ar[2][j] = xmax
                        all_ar[3][j] = ymax

                        for i1 in range(i,len(all_ar[1])-1):
                            all_ar[0][i1] = all_ar[0][i1+1]
                            all_ar[1][i1] = all_ar[1][i1+1]
                            all_ar[2][i1] = all_ar[2][i1+1]
                            all_ar[3][i1] = all_ar[3][i1+1]
                        all_ar[0].pop()
                        all_ar[1].pop()
                        all_ar[2].pop()
                        all_ar[3].pop()
                    else:
                        xmin = min(all_ar[0][i],all_ar[0][j])
                        ymin = min(all_ar[1][i],all_ar[1][j])
                        xmax = max(all_ar[2][i],all_ar[2][j])
                        ymax = max(all_ar[3][i],all_ar[3][j])

                        all_ar[0][i] = xmin
                        all_ar[1][i] = ymin
                        all_ar[2][i] = xmax
                        all_ar[3][i] = ymax

                        for i1 in range(j,len(all_ar[1])-1):
                            # print("aj")
                            all_ar[0][i1] = all_ar[0][i1+1]
                            all_ar[1][i1] = all_ar[1][i1+1]
                            all_ar[2][i1] = all_ar[2][i1+1]
                            all_ar[3][i1] = all_ar[3][i1+1]

                        all_ar[0].pop()
                        all_ar[1].pop()
                        all_ar[2].pop()
                        all_ar[3].pop()
                        break
                
                # print('intersection_area',i,j,intersection_area/(area1 + area2 - intersection_area))

    # 3 adding ' to below 
    for i in range(len(all_ar[1])):
        
        if i < len(all_ar[2]) and all_ar[3][i] < h*5/2 and i != 0:
            # print("ch",i,len(all_ar[3]))
            xmin = min(all_ar[0][i],all_ar[0][i-1])
            ymin = min(all_ar[1][i],all_ar[1][i-1])

            xmax = max(all_ar[2][i],all_ar[2][i-1])
            ymax = max(all_ar[3][i],all_ar[3][i-1])

            all_ar[0][i-1] = xmin
            all_ar[1][i-1] = ymin
            all_ar[2][i-1] = xmax
            all_ar[3][i-1] = ymax

            for i1 in range(i,len(all_ar[1])-1):
                all_ar[0][i1] = all_ar[0][i1+1]
                all_ar[1][i1] = all_ar[1][i1+1]
                all_ar[2][i1] = all_ar[2][i1+1]
                all_ar[3][i1] = all_ar[3][i1+1]
            all_ar[0].pop()
            all_ar[1].pop()
            all_ar[2].pop()
            all_ar[3].pop()
            # print("adding upper maatra",i)

    # 4 adding maatra to left side
    for i in range(len(all_ar[0])):
        # wi = all_ar[2][i] - all_ar[0][i]
        # print("I",i,w*3.5,all_ar[2][i] - all_ar[0][i])
        if (i < len(all_ar[0])) and  (all_ar[2][i] - all_ar[0][i]) < 46:
            # print("matra",i)
            if i != 0:
                xmin = min(all_ar[0][i],all_ar[0][i-1])
                ymin = min(all_ar[1][i],all_ar[1][i-1])
                xmax = max(all_ar[2][i],all_ar[2][i-1])
                ymax = max(all_ar[3][i],all_ar[3][i-1])

                all_ar[0][i-1] = xmin
                all_ar[1][i-1] = ymin
                all_ar[2][i-1] = xmax
                all_ar[3][i-1] = ymax

                for i1 in range(i,len(all_ar[1])-1):
                    all_ar[0][i1] = all_ar[0][i1+1]
                    all_ar[1][i1] = all_ar[1][i1+1]
                    all_ar[2][i1] = all_ar[2][i1+1]
                    all_ar[3][i1] = all_ar[3][i1+1]

                all_ar[0].pop()
                all_ar[1].pop()
                all_ar[2].pop()
                all_ar[3].pop()

    #4 adding thin to right index
    for i in range(len(all_ar[0])):
        if (i < len(all_ar[0])) and  (all_ar[2][i] - all_ar[0][i]) < 60:
            # print("thin",i)
            if i != (len(all_ar[0])-1):
                ymin = min(all_ar[1][i],all_ar[1][i+1])
                xmin = min(all_ar[0][i],all_ar[0][i+1])
                xmax = max(all_ar[2][i],all_ar[2][i+1])
                ymax = max(all_ar[3][i],all_ar[3][i+1])

                all_ar[0][i] = xmin
                all_ar[1][i] = ymin
                all_ar[2][i] = xmax
                all_ar[3][i] = ymax

                for i1 in range(i+1,len(all_ar[1])-1):
                    all_ar[0][i1] = all_ar[0][i1+1]
                    all_ar[1][i1] = all_ar[1][i1+1]
                    all_ar[2][i1] = all_ar[2][i1+1]
                    all_ar[3][i1] = all_ar[3][i1+1]

                all_ar[0].pop()
                all_ar[1].pop()
                all_ar[2].pop()
                all_ar[3].pop()

    #5 adding lower maatra to upper
    for i in range(len(all_ar[0])):
        if i != 0 and i < len(all_ar[0]) and all_ar[1][i] > int(h*5/2):

            xmin = min(all_ar[0][i],all_ar[0][i-1])
            ymin = min(all_ar[1][i],all_ar[1][i-1])
            xmax = max(all_ar[2][i],all_ar[2][i-1])
            ymax = max(all_ar[3][i],all_ar[3][i-1])

            all_ar[0][i-1] = xmin
            all_ar[1][i-1] = ymin
            all_ar[2][i-1] = xmax
            all_ar[3][i-1] = ymax

            for i1 in range(i+1,len(all_ar[1])-1):
                all_ar[0][i1] = all_ar[0][i1+1]
                all_ar[1][i1] = all_ar[1][i1+1]
                all_ar[2][i1] = all_ar[2][i1+1]
                all_ar[3][i1] = all_ar[3][i1+1]
                
            all_ar[0].pop()
            all_ar[1].pop()
            all_ar[2].pop()
            all_ar[3].pop()

            # print("added lower matra to upper",i)

    # print(wi/len(all_ar[0]),w*3.5)   
    co = 0
    wo = []
    for i in range(len(all_ar[0])): 
        co = co + 1
        s_im = img[int(all_ar[1][i]):int(all_ar[3][i]),int(all_ar[0][i]):int(all_ar[2][i])]
        
        s_im = cv2.bitwise_not(s_im)
        # cv2.imshow("before border",cv2.resize(s_im,(300,300)))

        s_im = cv2.copyMakeBorder(s_im,20,80,50,50,cv2.BORDER_CONSTANT,value=[0,0,0])
        # _,s_im = cv2.threshold(s_im,120,255,cv2.THRESH_BINARY)
        # cv2.imshow("after border",cv2.resize(s_im,(300,300)))
        # img2 = cv2.rectangle(img,(all_ar[0][i],all_ar[1][i]),(all_ar[2][i],all_ar[3][i]),(22,12,0),2)
        # img2 = cv2.putText(img2,str(i),(all_ar[0][i],all_ar[1][i]),2,2,(22,12,0))
        # cv2.imshow("contour",img2)
        # cr_im = img[int(all_ar[1][i]):int(all_ar[3][i]),int(all_ar[0][i]):int(all_ar[2][i])]
        # cr_im = cv2.bitwise_not(cr_im)
    
        # cv2.imshow("sd",s_im)
        # cv2.waitKey(0)
        # print("shape after c crop",s_im.shape)
        res_c = predict.pre(s_im)
        wo.append(l[res_c])
    # print('word',wo)
    
    return wo
        # img = cv2.line(img,(0,int(h*5/2)),(w,int(h*5/2)),(234,2,46),3)
    # print("shape",h,w)
    # img = cv2.bitwisse_not(img)
    # cv2.imshow("contour",img)
    # cv2.waitKey(0)