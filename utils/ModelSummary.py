from torchsummary import summary

def summarize(model, input_size):
    print(summary(model, input_size=input_size))