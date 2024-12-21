import zipfile
import multiprocessing as mp
import time


def get_number(file_title, path_1, path_2, counter, lock):
    '''
    Функция находит файлы в папках в архиве 1 и архиве 2 и возвращает папку 
    с номером и номер из файлов в этих папках
    '''
    with zipfile.ZipFile(path_1, "r") as archive_1:
        with archive_1.open(file_title) as file:
            path_to_number = file.readline().decode("utf-8").strip().replace("\\", "/")

    with zipfile.ZipFile(path_2, "r") as archive_2:
        with archive_2.open(path_to_number) as file:
            number = float(file.readline().decode("utf-8").strip())
            
    with lock:
        counter.value += number

    return number


def get_sum(path_1, path_2):
    '''
    Функция вычисляет сумму чисел, полученных из файлов
    расположенных в двух zip-архивах 
    с использованием многопроцессорной обработки
    '''
    manager = mp.Manager()
    counter = manager.Value("f", 0.0)
    lock = manager.Lock()

    with zipfile.ZipFile(path_1, "r") as archive_1:
        files_names = [item for item in archive_1.namelist() if item.endswith(".txt")]

    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        results = pool.starmap(
            get_number,
            [
                (file_name, path_1, path_2, counter, lock)
                for file_name in files_names
            ]
        )

    return sum(results), counter.value


if __name__ == "__main__":
    start_time = time.time()
    result = get_sum("./Innopolis_Projects/Homework_7/path_8_8.zip", 
                     "./Innopolis_Projects/Homework_7/recursive_challenge_8_8.zip")

    print(f"Сумма значений всех чисел во всех файлах: {result[0]}")
    print(f"Значение глобального счетчика: {result[1]}")
    print(f"Время выполнения программы: {int(time.time() - start_time)} сек")
