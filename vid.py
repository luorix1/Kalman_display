import numpy as np, cv2
#import request
import os
import webbrowser
import urllib

def vid(lock):
    lock.acquire()
    new = open("data_rt.txt", "w")

    f = "yt.mp4"
    #f = "https://10.144.172.37:8080/videofeed"

    TP = []
    TP.append((340,266))
    TP.append((402,265))
    TP.append((403,318))
    TP.append((338,320))
    TP.append((91,141))
    TP.append((626,144))
    TP.append((622,437))

    TP.append((93,434))
    TrackData = []
    for i in range(8):
        TrackData.append([])
    for i in range(8):
        TrackData[i].append(TP[i])
    #webbrowser.open(f)
    cap = cv2.VideoCapture(f)
    #cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
    delay = int(1000/fps)
    # 추적 경로를 그리기 위한 랜덤 색상
    color = np.random.randint(0,255,(200,3))
    print(color)

    lines = None  #추적 선을 그릴 이미지 저장 변수
    prevImg = None  # 이전 프레임 저장 변수
    # calcOpticalFlowPyrLK 중지 요건 설정

    termcriteria =  (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    frames = 0
    while cap.isOpened():
        frames += 1
        ret,frame = cap.read()
        if not ret:
            break
        img_draw = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 최초 프레임 경우
        if prevImg is None:
            print("Now on first Frame")
            prevImg = gray
            # 추적선 그릴 이미지를 프레임 크기에 맞게 생성
            lines = np.zeros_like(frame)
            # 추적 시작을 위한 코너 검출  ---①
            templist = []
            for i in range(8):
                templist.append([[float(TP[i][0]),float(TP[i][1])]])
            print("templist")
            print(templist)
            #templist = [[[448,262]],[[448,262]],[[448,262]],[[448,262]]]
            prevPt = cv2.goodFeaturesToTrack(prevImg, 200, 0.1, 10)
            for i in range(8):
                prevPt[i] = templist[i]


            #print("TYPE IS",type(prevPt))
            #prevPt = np.asarray(templist)
            print("CONVERTED TYPE IS",type(prevPt))
            #prevPt = templist
            print("** INFORMATION **")
            print(prevPt,end="yes")
            print()
        else:
            print("********************ETC*********************","now on",frames,"th frame ************************")
            nextImg = gray
            # 옵티컬 플로우로 다음 프레임의 코너점  찾기 ---②
            nextPt, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg, \
                                            prevPt, None, criteria=termcriteria)
            # 대응점이 있는 코너, 움직인 코너 선별 ---③
            prevMv = prevPt[status==1]
            nextMv = nextPt[status==1]
            for i,(p, n) in enumerate(zip(prevMv, nextMv)):
                px,py = p.ravel()
                nx,ny = n.ravel()

                if i == 7:
                    new.write(str(int(nx-px) // 50) + "    " + str(int(nx-px) // 50) + "    100\n")
                    #print(str(int(nx-px) // 50) + "    " + str(int(nx-px) // 50) + "    100\n")

        #cv2.imshow('OpticalFlow-LK', img_draw)
        key = cv2.waitKey(delay)
        if key == 27 : # Esc:종료
            break
        elif key == 8: # Backspace:추적 이력 지우기
            prevImg = None
    cv2.destroyAllWindows()
    cap.release()
    new.close()
    lock.release()