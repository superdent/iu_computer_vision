# Verkehrszeichenerkennung – Projektnotizen

## Stand: Vorbereitung, Datensatzaufbereitung und Training

### 1. Training
- Download GTSRB-Datensatz
- Aufbereitung des Datensatzes für die Verarbeitung mit Yolo -> Konvertierung in Yolo-Datei-Struktur
- Einrichtung Virtueller Environments für Yolov5 und Yolov8
- Download Yolov5 + Installation im V5-Environment
- Installation Yolov8 im V8-Environment

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
