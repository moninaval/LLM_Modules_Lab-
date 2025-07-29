import torch

def softmax_manual(x):
    # Step 1: subtract max for numerical stability
    x_stable = x - torch.max(x)
    
    # Step 2: exponentiate each element
    exp_x = torch.exp(x_stable)
    
    # Step 3: sum of exponentials
    sum_exp = torch.sum(exp_x)
    
    # Step 4: divide each exp by the sum
    return exp_x / sum_exp

if __name__ == "__main__":
    # Sample attention scores
    scores = torch.tensor([2.0, 1.0, 0.1])

    # Manual softmax
    manual_output = softmax_manual(scores)

    # PyTorch softmax
    torch_output = torch.softmax(scores, dim=0)

    # Print comparison
    print("Input scores:        ", scores)
    print("Manual softmax:      ", manual_output)
    print("Torch softmax:       ", torch_output)
    print("Difference:          ", torch.abs(manual_output - torch_output))
