import torch

class MyReLU(torch.autograd.Function):
    #it is possible to implement a customised autograd reinplementing the forward and backward functions
    @staticmethod
    def forward(ctx, input):
        # this function receive a tensor containing an input and return a tensor containing an output
        #by using save_for_backward I can cash arbitrary objects
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx,grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        return grad_input

dtype = torch.float
device = torch.device("cpu")

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

learning_rate = 1e-6
for t in range(500):
    relu = MyReLU.apply

    y_pred = relu(x.mm(w1)).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print(t,loss.item())

    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()

