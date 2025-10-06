# Verkehrszeichenerkennung
1. Verwendung des GTSRB-Datensatzes
1. Training verschiedener YOLO-Modelle
1. Videoanalyse mit trainierten Modellen
1. Verwendung eigender Bild-Daten

## 1. Ordnerstruktur anlegen
- training
- training/gtsrb für den Datensatz gtsrb und die Aufbereitung der Daten
- training/configs für die Trainingskonfiguration von Yolo
- training/requirements für Python requirements-Dateien
- training/scripts für Trainings-Startskripte
- Mehrere virtuelle Environments werden benötigt und sukzessive unterhalb von training angelegt

## 2. Verwendung des GTSRB-Datensatzes:
### Aufbau des Datensatzes
- GTSRB = **German Traffic Sign Recognition Benchmark**
- Der Datensatz enthält `.ppm`-Bilder, aufgeteilt in 43 Klassen (Ordner `00000` bis `00042`).
- Zu jedem Ordner gehört eine CSV-Datei mit Bounding Boxes und Klasseninformationen.
### Vorgehensweise
- Unterordner training/gtsrb
- Download des GTSRB-Datensatzes (GTSRB-Training_fixed) über https://benchmark.ini.rub.de/gtsrb_dataset.html#Downloads nach training/gtsrb/zip
- Virtuelles Environment für Datenaufbereitung anlegen (tools, requirements_tools.txt9)
- Erstellung passender Skripte für die Aufbereitung des Datensatzes zur Verarbeitung mit Yolo
- Kopieren/Verschieben der Quelldateien in das Unterverzeichnis training/gtsrb/ppm
- Yolo-Aufbereitung nach training/gtsrb/yolo_training
  - Konvertierung der `.ppm`-Bilder in `.jpg`
  - Erzeugung einer `.txt`-Datei pro Bild im YOLO-Format:  
    `class_id x_center y_center width height` (alle Werte relativ zur Bildgröße)
  - Die `.txt`-Datei trägt denselben Namen wie das zugehörige Bild (nur andere Endung).

```
python -m venv venv_tools
./venv_tools/scripts/activate 
pip install -r ./requirements/requirements_tools.txt 
python .\tools\ppm2yolo.py
```

## 3. Training
Die Beispiele unten beziehen sich auf eine Windows-Umgebung. Die Konfigurationsdateien gelten für alle Umgebungen.
Für Unix/Mac werden eigene Start-Scripte (Bash statt PowerShell) erstellt. 
- Unterordner "training"
- Erstellen passender Startskripte und Konfigurationsdateien unter training/scripts und training/configs
- Einrichtung Virtueller Environments für Yolov5 und Yolov8
- Download Yolov5 + Installation im V5-Environment
- Installation Yolov8 im V8-Environment

### Setup und Start des Trainings auf Windows:
#### Yolo V5
```
python -m venv venv-win-3.12-v5
git clone https://github.com/ultralytics/yolov5.git training/yolov5
.\venv-win-3.12-v5\Scripts\activate
pip install -r .\yolov5\requirements.txt
.\scripts\train_win_v5.ps1
```

#### Yolo V8 und V11
```
python -m venv venv-win-3.12-v8
.\venv-win-3.12-v8\Scripts\activate
.\scripts\train_win_v8.ps1
.\scripts\train_win_v11.ps1
```

### 5. Filterung auf ausgewählte Klassen
Das endgültige Training wurde mit ausgewählten Klassen durchgeführt.
- Reduzierung des Datensatzes auf 4 Klassen:
  - Klasse 14: **Stop**
  - Klasse 12: **Vorfahrt gewähren**
  - Klasse 13: **Vorfahrt**
  - Klasse 15: **Verbot der Einfahrt**
- Ein zusätzliches Skript filtert die Bild-/Labelpaare nach diesen Klassen.
- Zielverzeichnis: `datasets/yolo_training_filter/` mit YOLO-kompatibler Struktur

## 4. Videoanalyse
- Eine einstündige Dashcam-Fahrt bei Regen von Essen nach Köln wurde aufgezeichnet.
- Im Notebook wurden mit **OpenCV** einzelne Frames aus dem Video extrahiert.
- Ziel: Grundlagen verstehen (Frameverarbeitung, Bildformate, Timing).

## 5. Colab Git-Test 2