# SW Driverless – Cone Detection & Track Estimation

Realizzazione della task di recruiting per il team Eagle nell'area driverless della sezione software. 
I punti implementati sono i seguenti:

1. Load and Display Captured data
2. Cone detection
3. Object Classification
4. Extraction of Track Edges
5. Pose Estimation

I dettagli di come ogni punto è stato implementato, le difficoltà incontrate e come esse sono state superate e possibili miglioramenti sono reperibili all'interno del [report tecnico](doc/Technical_report.pdf).

---

## ⚙️ Requisiti

- **C++17** o superiore  
- **CMake >= 3.10**  
- **OpenCV >= 4.5**

---
## Setup/uso dell'applicazione
<pre> 
    ``` 
        mkdir build && cd build 
        cmake .. 
        make 
        ./cone_detector 
    ``` 
</pre>

Una volta compilato il procetto con il comando "make", apparirà un altro file, chiamato "./color_detection". Si tratta di un piccolo applicativo realizzato al volo e utilizzato per fare tuning sui colori, al fine di restringere il più possibile i range delle componenti di ogni cono e tagliare aree superflue per lo svolgimento dell'esercizio.










