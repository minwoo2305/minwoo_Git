class EarlyStopping():
    def __init__(self):
        self._step = 0
        self._loss = float('inf')
        self.loss_list = []
        self.flag = False

    def validate(self, loss, patience=0):
        if self._loss < loss:
            if self._step == patience:
                print("previous loss : " + str(self._loss) + ", current loss : " + str(loss))
                print('Training process is stopped early....')
                return 1
            else:
                self._step += 1
                return 0
        else:
            self._step = 0
            self._loss = loss
            return 0

    def validate_mean(self, loss, patience):
        if self.flag:
            if len(self.loss_list) != patience:
                self.loss_list.append(loss)
                return 0
            else:
                if sum(self.loss_list)/patience < loss:
                    print(str(patience) + " - Avg loss : " + str(sum(self.loss_list)/patience) + ", current loss : " + str(loss))
                    print('Training process is stopped early....')
                    return 1
                else:
                    self.loss_list.pop(0)
                    self.loss_list.append(loss)
                    return 0
        else:
            if self._loss < loss:
                self.flag = True
            else:
                self._loss = loss
                return 0

