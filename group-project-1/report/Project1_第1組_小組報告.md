% 多媒體技術與應用第一組 Project1 報告
% 109590011 陳彥宇、109590026 黃亮維

## 執行專案步驟

1. 執行 `video_clip.py` 將影片切成數張圖片
2. 用 LabelImg 標記圖片並以 YOLO 格式輸出
3. 用 `label_video.py` 將所有圖片合併成影片

## 執行程式遇到的困難

1. 檔案中讀取路徑的部分沒有寫好。
- 解法：將程式碼中有關路徑的部分都檢查，並更正。

2. 在 `label_video.py` 中讀取 LabelImg 的時候，沒有將文字處理好，導致出現 helmet 標示為 soup 的錯誤。
- 解法：更改 `classes.txt` 中的內容，比如 onion soup 改為 onion_soup。

## 選擇使用的資料集類別和影片

- 資料集類別：工人辨識
- 影片名稱：helmet1.mp4

## 程式執行的結果

- [txt 和 jpg 檔案連結（如果簡報的連結有問題請點 Youtube 影片說明欄裡的連結）](https://drive.google.com/drive/folders/11e8YmwBzLPeSzMajyIiXrHpdx3DWTvOk?usp=sharing)
- [Youtube 影片連結](https://youtu.be/4fnYtjRzzwA)
