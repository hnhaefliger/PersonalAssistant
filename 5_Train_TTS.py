from TTS import Train

if __name__ == '__main__':
    trainer = Train.Trainer('Dataset/TTS/dictionary.txt')

    history = trainer.train(5).history

    trainer.save('TTS/Models/model1.h5')
    
    with open('TTS/Models/model1history.txt', 'w+') as f:
        f.write(history)
