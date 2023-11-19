# PCG projekt 1
- autor: xpleva07

## Měření výkonu (čas / 100 kroků simulace)
### Průběžné
|   N   | CPU [s]  | Step 0 [s] | Step 1 [s] | Step 2 [s] |
|:-----:|----------|------------|------------|------------|
|  4096 | 0.492139 |  0.395949  |  0.241238  | 0.244879   |
|  8192 | 1.471328 |  0.792001  |  0.483616  | 0.488096   |
| 12288 | 2.478942 |  1.19003   |  0.726568  | 0.730627   |
| 16384 | 3.386801 |  1.585024  |  0.968202  | 0.973801   |
| 20480 | 5.059240 |  1.981085  |  1.210569  | 1.217129   |
| 24576 | 7.112179 |  2.377634  |  1.453834  | 1.460689   |
| 28672 | 9.892856 |  2.774522  |  1.695738  | 1.704007   |
| 32768 | 12.59829 |  3.168332  |  1.937347  | 1.947789   |
| 36864 | 15.54297 |  3.565476  |  2.181215  | 2.190890   |
| 40960 | 19.36099 |  3.960602  |  2.422557  | 2.435076   |
| 45056 | 23.48723 |  4.356295  |  2.664709  | 2.678488   |
| 49152 | 27.69359 |  4.752933  |  2.906905  | 2.921901   |
| 53248 | 32.63063 |  5.149035  |  3.149622  | 3.164875   |
| 57344 | 37.43660 |  9.067721  |  5.597071  | 5.458079   |
| 61440 | 42.85863 |  9.747650  |  6.026864  | 5.851553   |
| 65536 | 49.46104 |  10.418229 |  6.454878  | 6.241327   |
| 69632 | 55.14939 |  11.068397 |  6.857656  | 6.627901   |
| 73728 | 62.04446 |  11.722132 |  7.262365  | 7.020905   |
| 77824 | 69.26138 |  12.371849 |  7.663751  | 7.413508   |
| 81920 | 76.60071 |  13.024090 |  8.067749  | 7.798551   |

### Závěrečné

in 1 step
47 FPO for N * N
18 FPO for N

1float = 2B
steps  = 100

|    N   |  CPU [s] | GPU [s] | Zrychlení | Propustnost [GiB/s] | Výkon [GFLOPS] |
|:------:|:--------:|:-------:|:---------:|:-------------------:|:--------------:|
|   1024 |   1.0928 | 0.067993| 16x       | 145                 | 72.5           |
|   2048 |   0.5958 | 0.131091| 4.5x      | 300.8               | 150.4          |
|   4096 |   0.6652 | 0.253868| 2.5x      | 621.2               | 310.6          |
|   8192 |   1.6599 | 0.500523| 3.3x      | 1261.7              | 630.8          |
|  16384 |   3.3655 | 0.993888| 3.3x      | 2538.8              | 1269.4         |
|  32768 |  12.7233 | 1.981659| 6.5x      | 5093.3              | 2546.6         |
|  65536 |  48.9732 | 6.307309| 7.7x      | 6401.2              | 3200.6         |
| 131072 | 195.9965 |18.908289| 10.3x     | 8540.8              | 4270.4         |

## Otázky

### Krok 0: Základní implementace
**Vyskytla se nějaká anomále v naměřených časech? Pokud ano, vysvětlete:**
Pro N 53248 a 57344 došlo k velkému nárustu výpočetního času  9.067721 
Nejspíše se prodce zvedl počet kolizí v přístupu do GM.
### Krok 1: Sloučení kernelů
**Došlo ke zrychlení?**
Ano

**Popište hlavní důvody:**
Nyný není nutné čekat na sokončení kernelu a opětovné spuštení z CPU dále také byly eliminovány některé duplikované výpočty a cikly

### Krok 2: Sdílená paměť
**Došlo ke zrychlení?**
Ano ale drobnému

**Popište hlavní důvody:**
Jádra dřípve načítaly z globální paměti do lokálniho registru a to všechna stejnou hodnotu nyní načtou jednotlivá vlákna různé hodnoty do SM, která je narodíl od GM nizkolateční a přístup do ní je rychlejší ale není tak velká jako GM takže tam nejdou uložit všechny hodnoty a kvůli tomu je je nutné vyhazovat a obětovně načítat z GM což stojí rezijní čas.

### Krok 5: Měření výkonu
**Jakých jste dosáhli výsledků?**
GPU implementace je zhruba 10x rychleší na velkých datech u malých dat lze pozorovat jen malé zrychlení okolo 4x což je spusobeno velkou režijí na přeso dat tam a změt pro malém množství operací

**Lze v datech pozorovat nějaké anomálie?**
Ano na datech lze pozorovat manší odchilky až v řádech 1e-3 nicméně nejedná se o příliš velké odchilky aby neprošly testy
