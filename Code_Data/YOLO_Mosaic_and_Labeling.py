import cv2
import numpy as np
import os
import sys

def LoadNetwork(weights_path, config_path):
    
    net=cv2.dnn.readNet(weights_path,config_path)    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    return net,layer_names,output_layers

def LoadClass(names_path):
    
    classes=[]
    with open(names_path,"r") as f:
        classes=[line.strip() for line in f.readlines()]
    #colors=np.random.uniform(0,255,size=(len(classes),3))    
    return classes #,colors


def ImgLoad(image_path):
    
    img=cv2.imread(image_path)
    # print(img.shape)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    
    return img,blob,height, width, channels

def ReturnBoundingBox(outs,width,height):
    
    class_ids = []
    confidences = []
    boxes = []
    cent_boxes=[]
    for out in outs:
    
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # 검출 신뢰도
            if confidence > 0.5:
               # Object detected
               # 검출기의 경계상자 좌표는 0 ~ 1로 정규화되어있으므로 다시 전처리  
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                dw = int(detection[2] * width)
                dh = int(detection[3] * height)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                cent_boxes.append([center_x, center_y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # print(boxes)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)
    cent_indexes = cv2.dnn.NMSBoxes(cent_boxes,confidences, 0.45, 0.4)
    
    return class_ids, confidences, boxes, indexes, cent_boxes,cent_indexes

def label_file_name_make(image_path,train_data_dir):
    
    image_name=image_path.split(sep='/')[-1]
    label_name=image_name.replace('.jpg','.txt')
    
    return train_data_dir+'/'+label_name

            
def img_file_name_make(image_path,train_data_dir):
    image_name=image_path.split(sep='/')[-1]
    return train_data_dir+'/'+image_name



def Mosaic_And_Copy_Process(img,boxes,indexes, classes,class_ids,confidences,rate,img_height,img_weidth,cent_boxes,cent_indexes,image_path,train_data_dir): 
    
    count=0
    
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h=boxes[i]
            label=str(classes[class_ids[i]])
            score=confidences[i]
                
            if (label == 'biker') or (label == 'car') or (label == 'pedestrian') or (label == 'truck'):
                roi=img[y:y+h,x:x+w]
                roi_h,roi_w,_=roi.shape
                # print(type(roi.shape))
                # print(roi_h,roi_w)
                if roi_w > 0 and roi_h >0 :
                    roi=cv2.resize(roi,(roi_w,roi_h) , fx=1/rate, fy=1/rate, interpolation=cv2.INTER_NEAREST)
                    roi=cv2.resize(roi,(roi_w,roi_h),interpolation=cv2.INTER_NEAREST)
                    img[y:y+h,x:x+w]=roi
                    print("mosaic.....")
            elif label == 'pothole':
                
                count+=1
                label_file=label_file_name_make(image_path,train_data_dir)
                center_x,center_y,w,h=cent_boxes[i]
                file=open(label_file, mode='at', encoding='utf-8')
                file.write("{} {} {} {} {}\n".format(class_ids[i], center_x/img_weidth, center_y/img_height, w/img_weidth, h/img_height))
                file.close()
                
    if count>0:
        save_image_name=img_file_name_make(image_path,train_data_dir)
        cv2.imwrite(save_image_name,img)
    # return img





if __name__ == "__main__":
    img_dir = sys.argv[1] #
    weights_path=sys.argv[2]
    config_path=sys.argv[3]
    names_path=sys.argv[4]
    # yolo_data_path=sys.argv[5]
    train_data_dir=sys.argv[5]
    train_data_list_text_file=sys.argv[6]
    valid_data_list_text_file=sys.argv[7]
    newYoloData_path=sys.argv[8]
    newYoloBackupDir=sys.argv[9]
    
    model,layer_names,output_layers=LoadNetwork(weights_path,config_path)
    classes=LoadClass(names_path)    
    
    file_list=os.listdir(img_dir)
    for i in range(len(file_list)):
        image_path=img_dir+'/'+file_list[i]
        print(image_path)
        img,blob,h,w,c=ImgLoad(image_path)
        model.setInput(blob)
        outs = model.forward(output_layers)
        class_ids, confidences, boxes, indexes, cent_boxes,cent_indexes =ReturnBoundingBox(outs,w,h)
        Mosaic_And_Copy_Process(img,boxes,indexes, classes,class_ids,confidences,10,h,w,cent_boxes,cent_indexes,image_path,train_data_dir)
    
    train_file_list=os.listdir(train_data_dir)
    train_file=open(train_data_list_text_file, mode='at', encoding='utf-8')
    valid_file=open(valid_data_list_text_file, mode='at', encoding='utf-8')
    for i in range(len(train_file_list)):
        if '.jpg' in train_file_list[i] :
            if np.random.rand(1) < 0.8:
                train_file.write(train_data_dir+'/'+train_file_list[i]+'\n')
            else:
                valid_file.write(train_data_dir+'/'+train_file_list[i]+'\n')

    train_file.close()
    valid_file.close()
    
    new_yolo_data_file=open(newYoloData_path, mode='at', encoding='utf-8')
    new_yolo_data_file.write('classes = {}\n'.format(len(classes)))
    new_yolo_data_file.write('train = {}\n'.format(train_data_list_text_file))
    new_yolo_data_file.write('valid = {}\n'.format(valid_data_list_text_file))
    new_yolo_data_file.write('names = {}\n'.format(names_path))
    new_yolo_data_file.write('backup = {}'.format(newYoloBackupDir))
    new_yolo_data_file.close()
    
    print("complete.....")
    sys.exit()