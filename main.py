import numpy as np
from cfg import *

# Ağırlık dosyası eklemeniz gerekmektedir
# TODO#1: Görüntü seçme, dil seçme(c1 ve diğerlerini harflere dönüştürecek bir dataclass) ve confidence threshold argparse ile yapılacak
# TODO#2: Precision, recall ve f1-skor grafikleri çizdirilecek
# TODO#3: Üstünde etiketleme yapan ve etiketlenen resmin bilgilerini tutacak ayrı bir script hazırlanacak

IMG1 = "foto1.jpg"
IMG2 = "foto2.jpg"
selected = pic2

model = cv2.dnn.readNetFromDarknet("yolov4-obj.cfg", "yolov4-obj_last.weights")
layers = model.getLayerNames() # modeldeki katmanlar
unconnect = model.getUnconnectedOutLayers()
unconnect = unconnect - 1 #Output katmanları
box_label_dict = {}

output_layers = [] # Çıktı katmanlarını tutacak
for i in unconnect:
    output_layers.append(layers[int(i)])

classFile = 'obj.names'
with open(classFile, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')  # Sınıfları listeye aktarma
    for c in class_names:
        box_label_dict[c] = []

#######OKUNACAK RESMİ SEÇ##########
img = cv2.imread(IMG2)
img_truth = cv2.imread(IMG2) # Hem tahmin hem de gerçek kutucukların bulunduğu resim
###################################

img_width = img.shape[1]
img_height = img.shape[0]

img_blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True) # Fotoğrafı 416x416 yap (yolov4 416x416 kabul ediyor)

model.setInput(img_blob) #Resmi modele ver
detection_layers = model.forward(output_layers) # Obje tespiti yapıyoruz

ids_list = [] # Tespit edilen sınıfların idleri
boxes_list = [] # Tespit edilen sınıfların koordinatları
confidences_list = [] # Güvenilirlik skoru


for detection_layer in detection_layers:
    for object_detection in detection_layer:
        scores = object_detection[5:] # 0-3 koordinatlar, 5 ve sonrası tespit edilen cismin güvenilirlik skoru
        predicted_id = np.argmax(scores) # En yüksek skorlu id'yi seçer
        confidence = scores[predicted_id]

        if confidence > 0.50:
            label = class_names[predicted_id]
            bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
            start_x = int(box_center_x - (box_width / 2))
            start_y = int(box_center_y - (box_height / 2))

            ids_list.append(predicted_id) # Tahmin edilen sınıfları oluşturuyor
            confidences_list.append(float(confidence))
            boxes_list.append([start_x, start_y, int(box_width), int(box_height)]) # Kutuların koordinatları (skor> 0.10)

max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4) # Çakışan kutucukları eleyerek en optimize olanı seçer, 0.5 ve 0.4 eşik değerleri

# İŞARETLEDİĞİMİZ DOĞRU KUTULAR
for i in range(len(selected)):
    cv2.rectangle(img_truth, (selected[i][0], selected[i][1]), (selected[i][2], selected[i][3]),
                  (255, 0, 0), 2)

for max_id in max_ids: # En iyi kutuların  koordinatlarını tut, sınıflarına ayır
    max_class_id = max_id
    box = boxes_list[max_class_id]

    start_x = box[0]
    start_y = box[1]
    box_width = box[2]
    box_height = box[3]

    predicted_id = ids_list[max_class_id]
    label = class_names[predicted_id]
    confidence = confidences_list[max_class_id]

    end_x = start_x + box_width
    end_y = start_y + box_height

    box_label_dict[label].append([start_x, start_y, end_x, end_y]) # hangi sınıftansa o sınıfın listesine atıyor
    cv2.rectangle(img_truth, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2) # tahminleri çiziyor
    cv2.putText(img_truth, label, (start_x, start_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, 1)

# Doğru kutuları çizdiriyoruz asıl resme çizdiriyoruz
for i in range(len(selected)):
    cv2.rectangle(img_truth, (selected[i][0], selected[i][1]), (selected[i][2], selected[i][3]),
                  (0, 0, 255), 2)

box_labeled_and_predicted = {"BLUE": "Predicted", "RED": "Labeled"}
add_info(img_truth, infodict=box_labeled_and_predicted, margin=15, color=(255, 255, 255))

#cv2.imshow("img", img)
cv2.imshow("Image with truth boxes", img_truth)
cv2.waitKey(0)