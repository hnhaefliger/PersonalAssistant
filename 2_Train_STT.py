from STT import Train

if __name__ == '__main__':
    trainer = Train.Trainer('Dataset/STT')

    history = trainer.train(5).history

    trainer.save('STT/Models/model2.h5')
    
    with open('STT/Models/model2history.txt', 'w+') as f:
        f.write(history)
