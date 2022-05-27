import train
import test


def main():
    # model_name_list = ["resnext50", "densenet169"] # maq1
    # model_name_list = ["vgg16", "densenet169"] # maq3
    model_name_list = ["densenet121", "densenet169", "squeezenet", "resnext50", "vgg16"] # maq2
    # model_name_list = ["vgg16", "resnext50", "densenet169", "densenet121", "squeezenet"] # maq0
    
    for model_name in model_name_list:
        train.main(model_name)
        test.main(model_name)
    
    return 0


if __name__ == "__main__":
    main()