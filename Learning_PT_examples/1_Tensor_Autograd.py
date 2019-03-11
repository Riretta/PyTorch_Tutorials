import torch

dtype = torch.float
device = torch.device("cuda")

#N batch size
#D_in input dimension
#H hidden dimension
#D_out output dimension

N, D_in, H, D_out = 64, 1000, 100, 10

#random input and output
#we do not need to keep information about gradients with respect of this tensors
x = torch.randn(N, D_in, device= device, dtype= dtype)
y = torch.randn(N, D_out, device= device, dtype= dtype)

#random tensors for weights
w1 = torch.randn(D_in, H, device= device, dtype= dtype, requires_grad= True)
w2 = torch.randn(H, D_out, device= device, dtype= dtype, requires_grad= True)

learning_rate= 1e-6
for t in range(1500):
    #Forward
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    #loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # because we are using autograd we do not need to compute the backprop step manually
    # we can just call the backward from loss
    loss.backward()

    #the thing that we need to do manually is the update of weights of layers
    #that need to be torch.no_grad because we do not need to keep track of it
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        #zero the gradient!!
        w1.grad.zero_()
        w2.grad.zero_()
