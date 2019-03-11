import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

#we can define a model through the module function "sequencial". This function return a module that contains
#other modules and applies then in sequence to produce its output

model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out),
)

#in nn package there is defined also the most famous loss functions
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    #Forward
    y_pred = model(x)

    #loss
    loss = loss_fn(y_pred,y)
    print(t,loss.item())

    #zero gradient before running the gradient computation
    model.zero_grad()

    #Backward()
    loss.backward()

    #update weights
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate*param.grad