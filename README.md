# Verkehrszeichenerkennung
1. Verwendung des GTSRB-Datensatzes
1. Training verschiedener YOLO-Modelle
1. Videoanalyse mit trainierten Modellen
1. Verwendung eigender Bild-Daten

### 1. Ordnerstruktur anlegen
- training
- training/gtsrb für den Datensatz gtsrb und die Aufbereitung der Daten
- training/configs für die Trainingskonfiguration von Yolo
- training/requirements für Python requirements-Dateien
- training/scripts für Trainings-Startskripte
- Mehrere virtuelle Environments werden benötigt und sukzessive unterhalb von training angelegt

### 2. Verwendung des GTSRB-Datensatzes:
- Unterordner training/gtsrb
- Download des GTSRB-Datensatzes (GTSRB-Training_fixed) über https://benchmark.ini.rub.de/gtsrb_dataset.html#Downloads nach training/gtsrb/zip
- Virtuelles Environment für Datenaufbereitung anlegen
- Erstellung passender Skripte für die Aufbereitung des Datensatzes zur Verarbeitung mit Yolo
- Yolo-Aufbereitung nach training/gtsrb/yolo_training
```
pip install -r ./requirements requirements-tools.txt 
```

### 2. Training
- Unterordner "training"
- Einrichtung Virtueller Environments für Yolov5 und Yolov8
- Download Yolov5 + Installation im V5-Environment
- Installation Yolov8 im V8-Environment

#### Setup und Start des Trainings auf Windows:
```
python -m venv venv-win-3.12-v5
python -m venv venv-win-3.12-v8
git clone https://github.com/ultralytics/yolov5.git training/yolov5
.\venv-win-3.12-v5\Scripts\activate
pip install -r .\yolov5\requirements.txt
.\scripts\train_win_v5.ps1
```

### 1. Einrichtung des Projekts
- Es wurde ein lokales Python-Projekt angelegt.
- Ziel: Entwicklung eines Systems zur Verkehrszeichenerkennung in Videos.
- Innerhalb des Projekts wurde ein **Jupyter Notebook** erstellt, um erste Tests durchzuführen.

### 2. Erste Tests mit eigenem Video
- Eine einstündige Dashcam-Fahrt bei Regen von Essen nach Köln wurde aufgezeichnet.
- Im Notebook wurden mit **OpenCV** einzelne Frames aus dem Video extrahiert.
- Ziel: Grundlagen verstehen (Frameverarbeitung, Bildformate, Timing).

### 3. Verwendung eines Trainingsdatensatzes
- Da das eigene Videomaterial nicht annotiert ist, wurde ein öffentlich verfügbarer Datensatz verwendet.
- Entscheidung fiel auf den **German Traffic Sign Recognition Benchmark (GTSRB)**.
- Der Datensatz enthält `.ppm`-Bilder, aufgeteilt in 43 Klassen (Ordner `00000` bis `00042`).
- Zu jedem Ordner gehört eine CSV-Datei mit Bounding Boxes und Klasseninformationen.

### 4. Vorbereitung für YOLO
- Ziel war die Umwandlung des GTSRB-Datensatzes in ein YOLO-kompatibles Format:
  - Konvertierung der `.ppm`-Bilder in `.jpg`
  - Erzeugung einer `.txt`-Datei pro Bild im YOLO-Format:  
    `class_id x_center y_center width height` (alle Werte relativ zur Bildgröße)
- Die `.txt`-Datei trägt denselben Namen wie das zugehörige Bild (nur andere Endung).

### 5. Umsetzung mit Skript
- Zur Umwandlung wurde im Projekt ein Unterordner `tools/` angelegt.
- Darin befindet sich das Python-Skript `ppm2jpg.py` mit folgenden Funktionen:
  - Rekursive Verarbeitung ab einem definierten Eingabeverzeichnis
  - Konvertierung aller `.ppm`-Bilder in `.jpg`
  - Erstellung der zugehörigen YOLO-Labeldateien aus den CSV-Daten
  - Ausgabe erfolgt in einem separaten Zielverzeichnis mit gleicher Ordnerstruktur
  - Das Zielverzeichnis wird bei jedem Lauf automatisch geleert

### 6. Filterung auf ausgewählte Klassen
- Für das erste Training wurde der Datensatz auf vier Klassen reduziert:
  - Klasse 14: **Stop**
  - Klasse 12: **Vorfahrt gewähren**
  - Klasse 13: **Vorfahrt**
  - Klasse 15: **Verbot der Einfahrt**
- Ein zusätzliches Skript filtert die Bild-/Labelpaare nach diesen Klassen.
- Zielverzeichnis: `datasets/yolo_training_filter/` mit YOLO-kompatibler Struktur

### 7. YAML-Konfiguration für YOLO
- Für das Training wurde eine YOLO-Konfigurationsdatei `yolo_training_1.yaml` erstellt.
- Diese enthält **alle 43 Klassen**, nicht nur die vier genutzten:
  - So kann das Training auf andere Klassen erweitert werden, ohne die YAML-Datei zu ändern.
- Verzeichnispfade zu `train/images`, `val/images` sowie die vollständige Klassenliste sind enthalten.

### 8. Training
- Training erfolgt lokal mit YOLOv5 auf dem gefilterten Datensatz.
- Der Trainingsprozess wurde erfolgreich gestartet, nachdem alle Label-Klassen zur YAML-Datei gepasst haben.
- Probleme mit inkonsistenten Klassen-IDs konnten durch die vollständige Klassenliste behoben werden.
- Erste Experimente laufen mit dem `yolov5s.pt`-Modell.

### 9. Befehle
- source .venv_yolo/bin/activate
- python ./yolo_training_v/train.py --img 640 --batch 16 --epochs 50 --data ./yolo_training.yaml --weights yolov5s.pt --name verkehrszeichen-neu

- yolo detect train data=data.yaml model=yolov8s.pt imgsz=256 epochs=50 batch=8
