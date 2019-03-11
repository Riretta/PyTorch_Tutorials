#If there is no clue of what you want in nn.module then you can write it as you prefer
import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H , D_out):
        super(TwoLayerNet,self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H,D_out)

    def forward(self,x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4)

for t in range(500):
    y_pred = model(x)
    loss = criterion(y_pred,y)
    print(t,loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()