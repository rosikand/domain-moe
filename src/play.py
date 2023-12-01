import models 
import torch 
import pdb 


def get_moe_model():
    model = models.MixtureOfExperts(num_experts=6, num_classes=2)
    return model 



def main():
    model = get_moe_model()
    print(model)
    sample = torch.rand(1, 3, 32, 32)
    output = model(sample)
    pdb.set_trace()



if __name__ == "__main__":
    main()