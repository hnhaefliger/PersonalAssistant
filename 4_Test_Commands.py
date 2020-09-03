from Command import Commands, Functions

if __name__ == '__main__':
    handler = Commands.Handler('Command/Commands.txt')

    data = handler.match('what is the weather going to be like on wednesday in switzerland')

    print(eval('Functions.' + data))
