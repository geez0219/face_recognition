## Interview Project: Face recognition system

利用 https://github.com/deepinsight/insightface 的程式碼
* 建立一個人臉資料庫
    * 包含十個人(可以包含有你自己)的人臉資料庫 **=> ./img/targets**
    * 每個人可以只有一張照片，也可以有多張
* 建立一個人臉辨識的 application
    * 利用 webcam/video file 當輸入(兩個都需要)
    * 顯示即時視訊
    * 在人臉顯示 bounding box
    * bounding box 旁邊顯示身分 或者 不在資料庫中
* 建立一個 jupyter notebook
    * 建立測試圖片，不限張數，但總共有十個不同人臉  **=> ./img/test**
    * 五個人是在資料庫中的人(但與資料庫中的照片不同)
    * 五個人不在資料庫中
    * 針對以上的簡單測試，呈現你覺得應該要呈現的數據，並在 notebook 中呈現 **=> evaluation.ipynb**
    * 畫出 precision recall curve (調整 similarity threshold)(簡化起見，針對是否在資料庫中來二元判定即可)
    * 選擇一個你覺得合適於應用的 similarity threshold, 並說明理由


##  Environment Setup
```bash
pip install -r requirements.txt
```

## How to use
### webcam input
```
python face_recognition.py webcam -f ./img/targets
```
The annotation stream will play on the screen.

### video input
```
python face_recognition.py video -f ./img/targets -i <path_to_video>
```
The annotation video will play on the screen.

### image input
```
python face_recognition.py img -f ./img/targets -i <path_to_img_in> -o <path_to_img_out>
```
The annotation image will be saved at \<path_to_img_out\>

## Evaluation of the system
The result of the evaluation is in the jupyter notebook file `evaluation.ipynb`