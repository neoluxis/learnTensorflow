import tensorflow as tf


# 数据加载，分别从训练的数据集的文件夹和测试的文件夹中加载训练集和验证集
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # 加载训练集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # 加载测试集
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # 返回处理之后的训练集、验证集和类名
    return train_ds, val_ds, class_names


# 测试mobilenet准确率
def test_mobilenet():
    # 加载数据
    train_ds, val_ds, class_names = data_load("../data/vegetable_fruit/image_data",
                                              "../data/vegetable_fruit/test_image_data", 224, 224, 16)
    # 加载模型
    model = tf.keras.models.load_model("models/mobilenet_fv.h5")
    # model.summary()
    # 测试
    loss, accuracy = model.evaluate(val_ds)
    # 输出结果
    print('Mobilenet test accuracy :', accuracy)


# 测试cnn模型准确率
def test_cnn():
    # 加载数据集
    train_ds, val_ds, class_names = data_load("../data/vegetable_fruit/image_data",
                                              "../data/vegetable_fruit/test_image_data", 224, 224, 16)
    # 加载模型
    model = tf.keras.models.load_model("models/cnn_fv.h5")
    # model.summary()
    # 测试
    loss, accuracy = model.evaluate(val_ds)
    # 输出结果
    print('CNN test accuracy :', accuracy)


if __name__ == '__main__':
    test_mobilenet()
    test_cnn()
