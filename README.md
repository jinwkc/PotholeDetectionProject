# Pothole Detection Project(KITECH)
---
### In Code_Data Dir
1. backup 디렉토리 :  yolo model의 재학습이 일어나는 과정에 중간 및 최종 모델이 저장이 되는 디렉토리

2. img_dir 디렉토리 : 학습된 모델에서 다시 재학습을 위해서 pothole에 대해서 라벨링 및 다른 사물에 대해서 모자이크 처리가 될 이미지 파일의 저장소

3. train_data 디렉토리 : shell script에서 실행하는 파이썬 파일에 의해서 pothole에 대해서 저장되는 라벨링 파일(.txt) 및 모자이크 처리가 된 이미지의 저장소

4. yolo.names : 데이터의 클래스에 대해서 정의된 파일

5. nn.cfg : YOLOv3에 대해서 기본적인 네트워크 구조에 대해서 정의된 파일. 해당 파일에 epoch, batch size 등 여러 내용 포함. 테스트 환경인 RTX 2080Ti에 맞게 조절

6. nn_final.weights : 기존에 학습된 YOLOv3의 학습결과 모델

7. AutoTraining.sh : 라벨링 및 YOLO모델의 재학습 쉘 스크립트

8. ```./AutoTraining.sh ./YOLO_Mosaic_and_Labeling.py ./img_dir ./nn_final.weights ./nn.cfg ./yolo.names ./train_data ./train.txt ./valid.txt ./new_Yolo.data ./backup```<br>
위와 같은 명령어를 수행시, train_data 디렉토리에 pothole이 감지된 이미지에서 모자이크 처리된 이미지와 해당 pothole의 좌표에 대해서 정보가 있는 txt파일이 생성. 그리고 train.txt, valid.txt 및 new_Yolo.data 와 같은 파일이 생기면서 YOLO 모델에 학습에 필요한 파일이 같이 생성이 되면서 바로 모델의 재학습 과정에 진입.

9. 재학습 결과는 backup 디렉토리에 nn_final.weights 파일로 저장.
