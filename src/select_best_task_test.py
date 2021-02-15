# coding=utf-8

from Expreriment import SelectTask


def main():
    select_task = SelectTask('best_task.txt', 2500, 50)
    select_task.select_task()


if __name__ == '__main__':
    main()
