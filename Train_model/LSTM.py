import torch
from torch import nn
from torch.autograd import Variable
from Train_model import Data_Input


# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x


if __name__ == '__main__':
    # 读取数据，nd.array类型
    train_x = Data_Input.return_x_train_array().astype('float32')
    train_y = Data_Input.return_y_train_array()

    test_x = Data_Input.return_x_test_array().astype('float32')
    test_y = Data_Input.return_y_test_array()

# ----------------- train -------------------
INPUT_FEATURES_NUM = 561
OUTPUT_FEATURES_NUM = 1
train_x_tensor = train_x.reshape(-1, 1, INPUT_FEATURES_NUM)  # set batch size to 1
train_y_tensor = train_y.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batch size to 1

# transfer data to pytorch tensor
train_x_tensor = torch.from_numpy(train_x_tensor).cuda()
train_y_tensor = torch.from_numpy(train_y_tensor).cuda()

lstm_model = LstmRNN(INPUT_FEATURES_NUM, 20, output_size=OUTPUT_FEATURES_NUM, num_layers=1).cuda()  # 20 hidden units
print('LSTM model:', lstm_model)
print('model.parameters:', lstm_model.parameters)
print('train x tensor dimension:', Variable(train_x_tensor).size())

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

prev_loss = 1000
max_epochs = 2000

train_x_tensor = train_x_tensor.cuda()

for epoch in range(max_epochs):
    output = lstm_model(train_x_tensor).cuda()
    loss = criterion(output.float(), train_y_tensor.float()).cuda()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if loss < prev_loss:
        torch.save(lstm_model.state_dict(), '../lstm_model.pt')  # save model parameters to files
        prev_loss = loss

    if loss.item() < 1e-4:
        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
        print("The loss value is reached")
        break
    elif (epoch + 1) % 100 == 0:
        print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

# prediction on training dataset
pred_y_for_train = lstm_model(train_x_tensor)
pred_y_for_train = pred_y_for_train.view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()

# ----------------- test -------------------
lstm_model = lstm_model.eval()  # switch to testing model

# prediction on test dataset
test_x_tensor = test_x.reshape(-1, 1,
                               INPUT_FEATURES_NUM)
test_x_tensor = torch.from_numpy(test_x_tensor)  # 变为tensor
test_x_tensor = test_x_tensor.cuda()

pred_y_for_test = lstm_model(test_x_tensor).cuda()
pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()

loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(test_y))
print("test loss：", loss.item())
# 画图部分，暂时不用
"""# ----------------- plot -------------------
plt.figure()
plt.plot(t_for_training, train_y, 'b', label='y_trn')
plt.plot(t_for_training, pred_y_for_train, 'y--', label='pre_trn')

plt.plot(t_for_testing, test_y, 'k', label='y_tst')
plt.plot(t_for_testing, pred_y_for_test, 'm--', label='pre_tst')

plt.xlabel('t')
plt.ylabel('Vce')
plt.show()
"""
