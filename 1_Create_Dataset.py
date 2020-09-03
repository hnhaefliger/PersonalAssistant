from Data.Create import Creator

if __name__ == '__main__':
    creator = Creator.Creator('Dataset/sentences.txt', 'Dataset/STT')

    while True:
        creator.nextSentence()
