from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

# 加载数据
test = pd.read_csv("soluble_dataset/PSI_Biology_solubility_trainsetESM1b.csv")
x_test_label = test.iloc[:, 0]
x_test = test.iloc[:, 1:]

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(x_test)
# 获取唯一的标签类别
unique_labels = x_test_label.unique()
label_names = ['insoluble', 'soluble']
colors = [(65, 105, 225), (255, 166, 80)] # Set the desired colors for the labels
colors = [(r/255, g/255, b/255) for (r, g, b) in colors]
# 绘制 t-SNE 可视化结果
plt.figure(figsize=(8, 8))
for label in unique_labels:
    indices = x_test_label == label
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label_names[label], color=colors[label])
# plt.title('t-SNE Visualization of Plus Embedding features',  fontsize=15)
plt.legend(loc="lower right", prop={'weight': 'normal', 'size': 15})
plt.show()