---
urlcolor: blue
---

# 多媒體技術與應用第一組 Project3 加分題報告

## 小組分工表

- 109590011 陳彥宇：程式與報告撰寫 50%
- 109590026 黃亮維：報告撰寫 50%

## 執行專案步驟

1. 在 Colab 建立 Social LSTM 的環境
2. 訓練軌跡預測模型
3. 用訓練好的模型產生可視化結果

## 執行程式遇到的困難

### tensor 轉型問題
  
在訓練到第二個 epoch 的時候，都會出現 tensor 轉型的錯誤而中斷訓練，個人猜測是 Colab 環境與開發團隊使用環境不同所導致的。透過 Google 之後，我們將 `helper.py` 的第 73 行程式碼做以下修改就解決了問題。

```py
o_mux, o_muy, o_sx, o_sy, o_corr = mux.cpu()[0, :],\
                                   muy.cpu()[0, :],\
                                   sx.cpu()[0, :],\
                                   sy.cpu()[0, :],\
                                   corr.cpu()[0, :]
```

### 無法繼續從上次的訓練結果繼續訓練

在開始訓練時我們一直有一個疑問，就是如何從上次訓練的結果繼續訓練，畢竟我們是客家的 Colab 免費版，一定沒辦法一次就訓練完成。在看過簡報之後，雖然發現有 `--save_every` 這個參數可以設定每訓練幾次後可以儲存狀態，可是在遇到 tensor 轉型的錯誤中斷訓練後重新訓練，又沒有從上次的訓練結果繼續訓練。更令人失望的是，開發團隊沒有提供說明文件，於是我們開始了 trace source code 之旅。
  
首先我們先去看了 `train.py` 的程式碼，所幸的是，開發團隊對於每個參數的用途都有詳細的註解，不幸的是，還是沒有看到有關從上次的訓練結果繼續訓練的參數，所以我們又繼續 trace 下去。之後發現在 `train.py` 的確是會有每訓練完一個 epoch 就會儲存一次模型資訊的程式碼，但翻遍專案裡的所有原始碼，就是找不到有載入先前訓練模型資訊並繼續訓練的程式碼。這個專案有寫說是用 PyTorch 實作的，而我們剛好之前有自學過一點 PyTorch，於是就自行改了 `train.py` 的程式碼，解決了這個問題。

### 輸出可視化結果時報錯沒有 `adjustText` 這個套件
  
執行 `pip install adjustText`

### 輸出可視化結果的程式碼不會動作

在解決前一個問題後，輸出可視化結果 `visualize.py` 竟然不會輸出任何東西，沒有終端機的輸出訊息，沒有顯示圖表，沒有輸出檔案，也沒有報錯，什麼都沒有。

由於擔心只用免費版 Colab 會訓練不完，所以做了一些功課想要課金買 Colab Pro 或其他雲端運算平台。但後來找到了一個叫 Gradient 的運算平台，也有提供免費的方案，因此這次的加分題我們是在 Colab 與 Gradient 上輪流訓練，讓模型是我們人只要醒者就是在訓練的狀態，這也是我們能來得及將模型訓練完成的原因。

既然在 Colab 上無法輸出可視化結果，那就換個平台做。幸好在 Gradient 上是可以動作的，但在輸出的過程中還是遇到了一些問題。首先是 Gradient 的環境預設沒有提供 FFmpeg，但是執行 `visualize.py` 需要用到，否則會報沒有 FFmpeg 的錯誤。在安裝 FFmpeg 之後，又報 `ffmpeg: error while loading shared libraries: libGL.so.1: cannot open shared object file: No such file or directory` 這個錯誤，經過 Google 之後，只要安裝 `libgl1-mesa-glx` 就沒問題了。

後來我們有用在 Gradient 上的解決方法到 Colab 用，但還是無法解決 `visualize.py` 無法動作的問題。

> 註：如果在免費版 Gradient 上安裝 FFmpeg 過程中遇到需要與終端機互動才能安裝的問題，可以改用 [ffmpeg-colab](https://github.com/XniceCraft/ffmpeg-colab) 這個 repo 安裝

## 程式執行結果

- [程式執行結果檔案連結](https://drive.google.com/drive/folders/1Vijn1S0jgST3QR5OY-iUM4rqRxb0OpoV?usp=sharing)


## 個人心得（109590011 陳彥宇）

這次 Project 3 加分題的部份，是用 Social LSTM 來訓練一個預測軌跡的模型。這次的作業大概是遇過最多問題的一次，但所幸最後都有解決。在做加分題之前，我還在想要不要課金買 Colab Pro 或其他雲端運算資源，因為助教說會訓練很久，我有點擔心如果只用免費版會訓練不完，還好最後還是有把模型訓練完成，雖然真的花了不少的時間。

## 個人心得（109590026 黃亮維）

這此Project3加分題是由我組員撰寫的，而我也在後續了解如何運作，畢竟模型訓練嘛，需要大量的時間是必然的，而能不能儲存當前訓練進度也是一個重大的問題，這種時刻就顯現出我大Colab Pro並Colab破，而是有用的酷玩意兒，希望期末專案訓練的部分可以更加快速，而不是醒著開跑，跑到睡著，在睡醒再跑一次，實為辛苦。
