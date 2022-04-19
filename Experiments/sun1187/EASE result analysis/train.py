from trainer import runs
from preprocess import process
from inference import infer, save_ans

if __name__ == "__main__":
    num_k = 10

    data = process()
    model, X, check_x = runs(data)
    infer(model, num_k, X)
    save_ans(check_x)
    
