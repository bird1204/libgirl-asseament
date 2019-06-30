# libgirl Assessment

共分四題 linux, python, RoR, ML 知識：

1. **linux** : 
	* 題目 :
		* 安裝Linux (建議使用虛擬機器或docker, 但因docker設定較為複雜, 可依您對其熟悉程度自由選用)	 
		* 於Linux 環境中安裝 Conda
		* 於 conda 創建新環境並安裝 lgl , 安裝方式請詳: [lgl on GitHub](https://github.com/libgirlenterprise/lgl.git) ( lgl為本團隊為python開發的Launcher開源專案 )
	* 專案位置 : /miniconda/dockerfile
2. **python** : 
	* 題目 :
		* 修改lgl的程式碼 (使用git clone), 將lgl run (包含lgl install) 從使用conda改成使用pipenv, 並經測試 (可擇任意python專案)確定改後可正常運作
		* [lgl on GitHub](https://github.com/libgirlenterprise/lgl.git)
	* 專案位置 : /python/dockerfile
3. **RoR** : 此為現場測試
4. **ML** : 
	* 題目 : Explain VC dimension (with growth function and VC-entropy if possible) and explain the VC dimension of the ML algorithm you’re familiar with. 
	* 專案位置 : 無，見下面章節。

	
# VC dimension

VC dimension 是用來衡量 (binary) classifier 的能力的方式，最主要的用處是告訴我們 : 

1. 有沒有可能因為採用了過於複雜的模型，導致 overfitting，
2. 從 training error 衡量 classifier 的 accuracy 

也就是找到 traning data ( <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>E</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>i</mi>
      <mi>n</mi>
    </mrow>
  </msub>
</math> ), test data ( <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>E</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>o</mi>
      <mi>u</mi>
      <mi>t</mi>
    </mrow>
  </msub>
</math> ), model complexity 三者間的平衡 :

![](http://beader.me/mlnotebook/section2/images/model_complexity_curve.png)

也就是說：

````
test error ≤ training error + penalty(complexity)
````

接下來，我們嘗試更深入理解 VC dimension

### 先備名詞

* dichotomy : <math>
  <mrow>
    <mi>H</mi>
  </mrow>
</math> 作用於 <math>
  <mrow>
    <mi>D</mi>
  </mrow>
</math> 能夠有幾種不同的二元分類结果？

* growth function : <math>
  <mrow>
    <mi>H</mi>
  </mrow>
</math> 作用於 <math>
  <mrow>
    <mi>D</mi>
  </mrow>
</math> **最多**能有幾種不同二元分類结果？ 

* shatter : <math>
  <mrow>
    <mi>H</mi>
  </mrow>
</math> 作用於有 <math>
  <mrow>
    <mi>N</mi>
  </mrow>
</math> 個 inputs 的 <math>
  <mrow>
    <mi>D</mi>
  </mrow>
</math> 時，如果 dichotomy 的數量 等於 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mn>2</mn>
    <mi>N</mi>
  </msup>
</math> 時，稱為 shatter ( 有沒有辦法在所有可能性都把資料做分類的意思 )

* break point 則是 shatter 開始失敗的那一點，用圖片來說明：

![](http://www.ycc.idv.tw/media/MachineLearningFoundations/MachineLearningFoundations.007.jpeg)

請看圖片左半邊，其中：  

* N = 1 時，dichotomy = 2 且等於 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mn>2</mn>
    <mi>N</mi>
  </msup>
</math> = 2，因此 n = 1 時是 shatter
* N = 2 時，dichotomy = 4 且等於 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mn>2</mn>
    <mi>N</mi>
  </msup>
</math> = 4，因此 n = 2 時是 shatter
* N = 3 時，dichotomy = 8 且等於 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mn>2</mn>
    <mi>N</mi>
  </msup>
</math> = 8，因此 n = 3 時是 shatter
* N = 4 時，dichotomy = 14 但是**不等於** <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mn>2</mn>
    <mi>N</mi>
  </msup>
</math> = 16，因此 n = 3 時**不是 shatter，break point = 4**

接下來，看圖片的右半邊，Hoeffding's Inequality 被替代成右上角的公式，並且發現我們可以用 break point 來尋找 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>m</mi>
    <mrow>
      <mrow>
        <mi>H</mi>
      </mrow>
    </mrow>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>N</mi>
  <mo stretchy="false">)</mo>
</math> 的上限，而且 break point 越大，代表複雜度越高，我們會用 VC dimension 來說明複雜度


### VC dimension

首先，上定義 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>d</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>V</mi>
      <mi>C</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <mi>B</mi>
  <mi>r</mi>
  <mi>e</mi>
  <mi>a</mi>
  <mi>k</mi>
  <mi>P</mi>
  <mi>o</mi>
  <mi>i</mi>
  <mi>n</mi>
  <mi>t</mi>
  <mo>&#x2212;<!-- − --></mo>
  <mn>1</mn>
</math> ：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mi>m</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mrow class="MJX-TeXAtom-ORD">
        <mi mathvariant="double-struck">H</mi>
      </mrow>
    </mrow>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>n</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <msup>
    <mi>n</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <msub>
        <mi>d</mi>
        <mrow class="MJX-TeXAtom-ORD">
          <mi>V</mi>
          <mi>C</mi>
        </mrow>
      </msub>
    </mrow>
  </msup>
  <mo>,</mo>
  <mtext>&#xA0;</mtext>
  <msub>
    <mi>d</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>V</mi>
      <mi>C</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <mi>B</mi>
  <mi>r</mi>
  <mi>e</mi>
  <mi>a</mi>
  <mi>k</mi>
  <mi>P</mi>
  <mi>o</mi>
  <mi>i</mi>
  <mi>n</mi>
  <mi>t</mi>
  <mo>&#x2212;<!-- − --></mo>
  <mn>1</mn>
</math>

也就是說，只要有 break point 存在，VC dimension就是一個有限的值，也因此growth function 是一個有限的值，VC bound 就產生了，可以確保 Bad Data 出現的機率被壓在一個定值之下，所以一樣的只要資料量 <math>
  <mi>N</mi>
</math> 夠多就可以確保 <math>
  <msub>
    <mi>E</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>i</mi>
      <mi>n</mi>
    </mrow>
  </msub>
  <mo>&#x2248;</mo>
  <msub>
    <mi>E</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>o</mi>
      <mi>u</mi>
      <mi>t</mi>
    </mrow>
  </msub>
</math> ，機器將可以學習。

### 常見演算法的 VC dimension

* d-D PLA -> d + 1
* d-D linear classifier -> d + 1
* d-node decision tree -> d + 1
* radial basis function classifier -> 無限大

````
在物理意義上，有幾個旋鈕，VC dimension 就是多少
````

