from data_loader import load_data, preprocess_data
from model import create_model
from train import train_model

if __name__ == "__main__":
    images_list = []  
    labels_list = [] 
    img_path = '/content/hirise-map-proj-v3/map-proj-v3/' 

    with open('/content/labels-map-proj.txt') as labels:
        for l in labels:
            file_name , label = l.split(' ')
            images_list.append(img_path + file_name)
            labels_list.append(int(label))
            
    images, labels = load_data(images_list, labels_list, img_path)
    train_images, test_images, train_labels, test_labels = preprocess_data(images, labels)
    
    model = create_model()
    
    BATCH_SIZE = 32
    num_epochs = 15
    train_model(model, train_images, train_labels, BATCH_SIZE, num_epochs)
    
    model.save('model.h5')
