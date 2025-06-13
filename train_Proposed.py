# train_Proposed.py (New file)

from trainer.Trainer_Proposed import Trainer_Proposed
from datetime import datetime

def main():
    trainer_proposed = Trainer_Proposed()
    trainer_proposed.train()

if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')