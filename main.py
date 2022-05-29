import cnn
import knn

knn_model = knn.KNN(1)

cnn_model = cnn.CNN(3, 256, 10, False)
cnn_model.visualize_accuracy_according_to_epoch()
cnn_model.visualize_loss_according_to_epoch()
cnn_model.visualize_label_distribution()
