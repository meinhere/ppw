---
title: Word Embedding

---

# Word Embedding
Representasi kata dalam sebuah dokumen dalam bentuk vektor yang fungsinya dapat menangkap konteks kata, hubungan antar kata, dan makna dari kata tersebut.

## Skip-Gram
Cara kerja dari Skip-gram adalah dengan **membaca adanya kata target, maka dilakukan prediksi kata konteks di sekitarnya**. Kata konteks didefinisikan sebagai kata yang mengelilingi kata target dalam ukuran kalimat yang tetap. Sehingga, akan dibuatkan sampel pelatihan dengan menggeser corpus teks dan membentuk pasangan (target, konteks).

### 1. Representasi setiap kata ke dalam bentuk vektor
![image](https://hackmd.io/_uploads/rkPDoUzR0.png)

Pada gambar diatas dapat dilihat bahwa kata "Achieve" dan "Success" sudah memiliki bentuk vektor (dalam gambar diberikan vektor dengan 3 elemen). Korelasi antara kata "Achieve" dan "Success" dapat diukur dari jarak antara kedua vektor.

Dibawah ini contoh analogi penggunaan kedekatan antara kata dalam bentuk vektor. Terdapat 3 kata yang digunakan yaitu "Indonesia", "ibukota", "Jakarta"

\begin{align*} 
vec(\text{ibukota}) & = [3.02 \quad -0.93 \quad 1.82] \\
vec(\text{Indonesia}) & = [1.22 \quad 0.34 \quad -3.82] \\ 
vec(\text{Jakarta})  & = [4.09 \quad -0.58 \quad 2.01]
\end{align*}

Untuk mengetahui vektor ibukota Indonesia, maka vektor kata "ibukota" dapat dijumlahkan dengan vektor kata "Indonesia".

\begin{align*}
    vec(\text{ibukota}) + vec(\text{Indonesia}) &= [3.02 \quad -0.93 \quad 1.82] + [1.22 \quad 0.34 \quad -3.82] \\
                                              &= [4.24 \quad -0.59 \quad -2.00]
\end{align*}

Setelah dilakukan proses penjumlahan, maka didapatkan hasil bahwa vektor kata "ibukota" dan vektor kata "Indonesia" itu berdekatan dengan vektor kata "Jakarta". Sehingga dapat disimpulkan bahwa ibukota dari Indonesia adalah Jakarta.

\begin{align*} 
[4.24 \quad -0.59 \quad -2.00] & \cong [4.09 \quad -0.58 \quad 2.01] \\ 
vec(\text{ibukota}) + vec(\text{Indonesia}) & \cong vec(\text{Jakarta})
\end{align*}


### 2. Windows size C dalam Skip-Gram
Windows size dalam Skip-Gram digunakan untuk menentukan jumlah kata konteks yang akan diambil dari kata target. Jika windows size $C = 1$, maka akan diambil satu kata disebelah kiri dan satu kata disebelah kanan dari kata target.

![image](https://hackmd.io/_uploads/BJxR4pAzCR.png)

Contoh:
**Kata Target** = passes
**Kata Konteks** = (pass, who) & (pass, the)

### 3. Struktur Skip-Gram
![image](https://hackmd.io/_uploads/BJ5g7DM0A.png)

Dengan menggunakan library Word2Vec kita dapat mengonversi setiap kata yang dipilih menjadi vektor. Setiap kata memiliki jumlah dimensi vektor yang telah ditentukan. Jika kita menentukan size = 3 maka kita mengisi nilai dari $N = 3$. Nilai dari $N$ dapat kita isi sesuaikan dengan kebutuhan ukuran elemen pada vektor yang diinginkan.

Corpus :
**"the man who passes the sentence should swing the sword"**

```python=
from gensim.models import Word2Vec

model = Word2Vec(corpus, size=3, window=1)
```

Hasil dari kode diatas berarti matriks bobot masukan ($W_{input}$) akan memiliki ukuran $8 × 3$, dan matriks bobot keluaran ($W_{output}^T$) akan memiliki ukuran $3 × 8$. Nilai $8$ didapat dari jumlah kata unik yang ada pada corpus **"the man who passes sentence should swing sword"** ($V = 8$) dan Nilai $3$ didapat dari nilai $N$.

#### 3.1 Forward Propagation: One Hot Encoding (Input Layer)
Input Layer disini ialah bentuk dari $V-$dim one hot encoding vektor. Dimana setiap kata yang berada pada nilai input akan diisikan menjadi $1$ selain itu akan diisikan dengan nilai $0$. Jumlah kolom yang ada disesuaikan dengan kemunculan kata unik pada corpus. Sehingga hasilnya sebagai berikut.


|    | the | man | who  | passes | sentence | should | swing | sword |
| --     | -- | -- | -- | -- | -- | -- | -- | -- |
| the     | 1     | 0     | 0     | 0 | 0 | 0 | 0 | 0 |
| man     | 0     | 1     | 0     | 0 | 0 | 0 | 0 | 0 |
| who     | 0     | 0     | 1     | 0 | 0 | 0 | 0 | 0 |
| passes  | 0     | 0     | 0     | 1 | 0 | 0 | 0 | 0 |
| the     | 1     | 0     | 0     | 0 | 0 | 0 | 0 | 0 |
| sentence| 0     | 0     | 0     | 0 | 1 | 0 | 0 | 0 |
| should  | 0     | 0     | 0     | 0 | 0 | 1 | 0 | 0 |
| swing   | 0     | 0     | 0     | 0 | 0 | 0 | 1 | 0 |
| the     | 1     | 0     | 0     | 0 | 0 | 0 | 0 | 0 |
| sword   | 0     | 0     | 0     | 0 | 0 | 0 | 0 | 1 |

Dengan bentuk one hot encoding vektor diatas, kita tidak bisa melihat korelasi atau kemiripan dari setiap kata. Sehingga perlu dilakukan proses perubahan nilai vektor yang dapat diukur korelasi antar kata.


#### 3.2 Forward Propagation: Bentuk Pasangan (target, konteks)
Diperlukan sebuah input (konteks) dan output (target) dari bentuk Skip-Gram yang akan dibuat. Dibawah ini adalah hasil dari target dan konteks dari corpus.

| No | Target | Konteks |
| -------- | -------- | -------- |
| 1     | the     | man     |
| 2     | man     | the     |
| 3     | man     | who     |
| 4     | who     | man     |
| 5     | who     | passes     |
| 6     | passes     | who     |
| 7     | passes     | the     |
| 8     | the     | passes     |
| 9     | the     | sentence     |
| 10     | sentence     | the     |
| 11     | sentence     | should     |
| 12     | should     | sentence     |
| 13     | should     | swing     |
| 14     | swing     | should     |
| 15     | swing     | the     |
| 16     | the     | swing     |
| 17     | the     | sword     |
| 18     | sword     | the     |

#### 3.3 Forward Propagation: Matriks Bobot Input dan Output ($W_{input}$, $W_{output}$)
Dari bentuk one hot encoding yang diatas, kita akan mengubah dalam bentuk vektor yang dapat diukur kedekatan antar kata dengan menggunakan Word2Vec untuk mendapatkan $W_{input}$ dan $W_{output}$. 

![image](https://hackmd.io/_uploads/B1X6meQR0.png)


Pada gambar diatas bentuk vektor dari kata **"passes"** menjadi $[0.1 \quad 0.2 \quad 0.7]$ dan vektor kata **"should"** menjadi $[-2 \quad 0.2 \quad 0.8]$. Dikarenakan menggunakan N = 3 maka hasil dari vektor yang dibentuk menjadi 3 elemen saja.

![image](https://hackmd.io/_uploads/S11I4xm0A.png)


#### 3.4 Forward Propagation: Hidden (Projection) Layer ($h$)
Skip-Gram menggunakan konsep neural network yang menggunakan satu hidden layer. Dalam konsep Natural Language Prosessing hidden layer dapat disebut sebagai projection layer. Dikarenakan $h$ itu sebenarnya $N-$dim projected vector dari one hot encoding dari input vektor

![image](https://hackmd.io/_uploads/r1mBBx7RC.png)

$h$ diperoleh dari proses mengalikan matriks bobot input layer ($W_{input}$) dengan $V-$dim input vektor.

\begin{align*}
h = W_{input}^T \cdot x  \in \mathbb{R}^{N}$
\end{align*}

#### 3.5 Forward Propagation: Softmax Output Layer
Output layer berisi probabilitas dari semua kata unik dalam corpus (${V-}$dim). Untuk menentukan nilai probabilitas diperlukan kata konteks dan kata target, contohnya jika ingin mengukur probabilitas kata $A$ dari kata target $B$ maka dapat dinotasikan sebagai $p(A|B)$. Pada Skip-Gram kita dapat menggunakan notasi $p(w_{context}| w_{center})$. Untuk mencari probabilitas kata konteks dengan kata target, maka diperlukan fungsi aktivasi softmax.

\begin{align*}
p(w_{context}|w_{center}) = \frac{exp(W_{output_{(context)}} \cdot h)}{\sum^V_{i=1}exp(W_{output_{(i)}} \cdot h)} \in \mathbb{R}^{1}
\end{align*}

* $W_{output_{(i)}}$ adalah nilai elemen vektor output dari kata ke-${i}$ yang memiliki ukuran $1 \times N$
* $W_{output_{context}}$ adalah nilai elemen vektor output dari kata konteks yang juga memiliki ukuran $1 \times N$
* $V$ adalah total kata unik dari corpus
* $h$ adalah hidden (projection) layer yang memiliki ukuran ($N \times 1$)
* Output yang dihasilkan adalah nilai skalar dengan ukuran ($1 \times 1$) yang memiliki probabilitas dengan rentang $[0, 1]$

Proses ini akan melakukan perulangan sejumlah $V$ kali sesuai dengan jumlah kata unik dalam corpus dengan kata target.

\begin{align*}
\left[\begin{array}{c} p(w_{1}|w_{center}) \\ p(w_{2}|w_{center}) \\ p(w_{3}|w_{center}) \\ \vdots \\ p(w_{V}|w_{center}) \end{array} \right] = \frac{exp(W_{output} \cdot h)}{\sum^V_{i=1}exp(W_{output_{(i)}} \cdot h)} \in \mathbb{R}^{V}
\end{align*}

Sehingga $W_{output}$ yang memiliki ukuran ($V \times N$) akan dikalikan dengan $h$ yang memiliki ukuran ($N \times 1$) sehingga menghasilkan dot product vector yang memiliki ukuran ($V \times 1$). Terakhir, dot product vector akan dimasukkan ke dalam fungsi aktivasi softmax.

![image](https://hackmd.io/_uploads/H1j9pt7C0.png)

Hasil nilai dari aktivasi softmax jika dijumlahkan semua akan bernilai $1$.

#### 3.6 Backward Propagation: Prediction Error
Skip-Gram model melakukan optimasi bobot matriks ($\theta$) dengan mengurangi prediction error. Prediction error merupakan perbedaan antara hasil dari softmax output layer ($y_{pred}$) dengan probabilitas target yang sebenarnya ($y_{true}$) dari setiap kata. $y_{true}$ merupakan vektor one hot encoder dari setiap kata konteks.

![image](https://hackmd.io/_uploads/H1-Q-bSC0.png)

Pada gambar di atas, terdapat 2 kata yaitu **"the"** dan **"who"** yang telah memiliki hasil dari pengurangan $y_{pred}$ dan $y_{true}$. Selanjutnya dilakukan penjumlahan untuk melihat bobot yang kemudian akan dimasukkan/diupdate pada matriks bobot yang ada.

![image](https://hackmd.io/_uploads/HJSKfWSCC.png)

Update matriks bobot terus dilakukan hingga mendapatkan prediction error paling kecil.

![image](https://hackmd.io/_uploads/rkY3M-rA0.png)


### 4. Demonstrasi Perhitungan

#### 4.1 Mencari nilai hidden layer ($h$)
Dalam contoh ini kita akan menggunakan kata target **"passes"** dengan nilai windows size $C = 1$, sehingga mendapatkan kata konteks yaitu **"the"** dan **"who"**. Selanjutnya dicarilah nilai dari hidden layer ($h$) dari matriks bobot input ($W_{input}$).

![image](https://hackmd.io/_uploads/Hk3FTlr0C.png)

#### 4.2 Menghitung output layer dengan fungsi aktivasi softmax

Pertama perlu dilakukan perkalian antara hidden layer ($h$) dengan matriks output layer ($W_{output}$). Setelah itu baru dilakukan proses perhitungan dengan fungsi aktivasi softmax.

![image](https://hackmd.io/_uploads/SybWJZS0C.png)

#### 4.3 Menjumlahkan prediction error dari kata konteks
Berhubung windows size $C = 1$ maka hanya terdapat 2 nilai error yang akan dijumlahkan yaitu nilai error dari kata **"the"** dan **"who"**.

![image](https://hackmd.io/_uploads/rkEVn4SRR.png)


#### 4.4 Mengitung $\nabla W_{input}$
Diperlukan rumus ($\frac{\partial J}{\partial W_{input}}$) yang dapat direpresentasikan juga dengan cara mengalikan vektor one hot encoding dari input layer ($x$) dengan $W_{output}^T \sum^C_{c=1} e_c$. Rumus $W_{output}^T \sum^C_{c=1} e_c$ merupakan hasil perkalian antara ($W_{output}$) dengan penjumlahan prediction error ($\sum^C_{c=1} e_c$).

\begin{align*}
\frac{\partial J}{\partial W_{input}}  = x \cdot (W_{output}^T \sum^C_{c=1} e_c)
\end{align*}

![image](https://hackmd.io/_uploads/r18zAErR0.png)


#### 4.5 Menghitung $\nabla W_{output}$
Diperlukan rumus ($\frac{\partial J}{\partial W_{output}}$) yang dapat direprenstasikan juga dengan cara mengalikan hidden layer ($h$) dengan penjumlahan prediction error ($\sum^C_{c=1} e_c$). Tidak sama dengan matriks bobot input ($W_{input}$) yang hanya mengupdate satu nilai vektor, tetapi semua nilai vektor pada matriks bobot output ($W_{output}$) akan diperbarui.

\begin{align*}
\frac{\partial J}{\partial W_{output}} = h \cdot \sum^C_{c=1} e_c
\end{align*}


![image](https://hackmd.io/_uploads/HyYJbSB00.png)


#### 4.6 Memperbarui matriks bobot input dan output
Matriks bobot input dan output akan diperbarui menggunakan rumus 

\begin{align*}
W_{input}^{(new)}=W_{input}^{(old)}- \eta \cdot x \cdot (W_{output}^T \sum^C_{c=1} e_c)
\end{align*}

![image](https://hackmd.io/_uploads/BkiRWHr0C.png)

\begin{align*}
W_{output}^{(new)}=W_{output}^{(old)}- \eta \cdot h \cdot \sum^C_{c=1} e_c
\end{align*}

![image](https://hackmd.io/_uploads/HyWyMSrCA.png)

