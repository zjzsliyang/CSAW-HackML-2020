import strip
import sys

if __name__ == '__main__':
    
    model_path = str(sys.argv[1])
    data_path = str(sys.argv[2])
    
    pred = strip.G(model_path, data_path)

    x, y = strip.load_data(data_path)
    
    print(pred)
    print(y)