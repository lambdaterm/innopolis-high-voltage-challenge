В этом репозитории содержится код команды "да не умер он в конце драйва",
подготовленный в рамках хакатона https://www.kaggle.com/competitions/innopolis-high-voltage-challenge


Инструкция по установке
-----------------------

В качестве архитектуры модели обнаружения пропусков в гирляндах изоляторов используются
наработки проекта Gold-YOLO (https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO).
В этом проекте был модифицирован исходный код некоторых модулей в пакете `ultralytics`.
Для того, чтобы упростить работу с репозиторием, файл `conv.py` (который лежит в корне)
подменяет родной `conv.py` пакета `ultralytics.nn.modules.conv` простым копированием.

Процедура установки зависимостей в виртуальное окружение следующая:

```commandline
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 example.py
```

Структура репозитория
---------------------

В каталоге `src/model` находятся веса всех моделей, составляющих пайплайн обработки.
`src/utils` содержит весь основной код: пре- и постпроцессинг, построение пайплайна обработки,
работа с разметкой CVAT.

Получение предиктов
-------------------

Для получения предсказаний ансамбля сетей служит скрипт `src/main.py`, который ожидает на входе
один параметр: путь до директории с исходными изображениями.
Пример запуска:
```shell
cd src
python3 main.py --indir input_dir
```

По завершении работы `src/main.py` в директории `src` появится файл `result.csv` с сериализованными
предиктами. По структуре этот файл повторяет файл сабмитов для kaggle.com
В `src/OUTPUT` сохраняются входные файлы, с отрисованными поверх прямоугольниками в местах пропуска
изоляторов в гирлянде.
