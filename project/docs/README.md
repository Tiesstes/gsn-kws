# Keyword spotting z implementacją embedding'u wektora mówców

Fuzja prac: https://arxiv.org/abs/2106.04140 oraz https://arxiv.org/html/2403.07802v1

prace pomocnicze/uzupełniające: 
- SSN: https://arxiv.org/abs/2103.13620
- Swish/SiLU: https://arxiv.org/abs/1710.05941v1

### Budowa bloków BC-ResNet:
![img.png](img.png)

**Występują 2 typy bloków: *przejściowe* oraz *normalne*.**
Główna różnica polega na tym, że jeden zmienia ilość kanałów wyjściowych, a drugi nie. 
Ich wizualizacja powyżej. Trik polega na skorzystaniu z point-wise convolution 
na początku bloku *przejściowego*. Tym samym, nieobecne jest połączenie rezydualne podające
czysty *x* do końcowego argumentu dla aktywacji - jak w przypadku bloku *normalnego*.

### Schemat architektury BC-ResNet-1:
![img_1.png](img_1.png)

### Embedding mówcy

#### Backbone: BC-ResNet-1
#### Fuse point: addition of user embeddings

![img_2.png](img_2.png)

### Postęp prac

1. Została stworzona klasa wrappująca *torchaudio.datasets SPEECHCOMMANDS*, która odpowiada za weryfikację
długości próbek, balans klas oraz zamianę sygnału z dziedziny czasu na dziedzinę częstotliwości (LogMel-Spektrogram).


2. Udało się zaimplementować BC-ResNet-1 'from PyTorch scratch'. Wchodzą w to: bloki BCResBlock, ConvBNReLU
oraz normalizacja podpsamami częstotliwości tzw. Subspectral Normalisation.


3. Dokonano fuzji map cech z BC-ResNet-1 z warstwą embedding'u przechowującą wektory 
reprezentujące cechy mowy dla każdego z mówców.

4. Zastosowano klasyczną funkcję straty dla zagadnień klasyfikacji: Entropię Skrośną. 