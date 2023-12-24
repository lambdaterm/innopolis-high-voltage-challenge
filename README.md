В этом репозитории содержится код команды "да не умер он в конце драйва",
подготовленный в рамках хакатона https://www.kaggle.com/competitions/innopolis-high-voltage-challenge


Инструкция по установке
-----------------------

В качестве архитектуры модели обнаружения пропусков в гирляндах изоляторов используются
наработки проекта Gold-YOLO (https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO).
В этом проекте был модифицирован исходный код некоторых модулей в пакете `ultralytics`.
Для того, чтобы упростить работу с репозиторием, файл `conv.py` (который лежит в корне)
подменяет родной `conv.py` пакета `ultralytics.nn.modules.conv.py` простым копированием.

Процедура установки зависимостей в виртуальное окружение следующая:

```commandline
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 example.py
```
